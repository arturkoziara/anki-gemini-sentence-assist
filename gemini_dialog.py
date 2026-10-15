"""Tools → Gemini Sentence Assist… — step wizard: Setup → Find → Generate."""

from __future__ import annotations

import time
from pathlib import Path

from aqt import mw
from aqt.qt import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    Qt,
    QTextEdit,
    QTimer,
    QVBoxLayout,
    QWidget,
    qconnect,
)
from aqt.utils import showInfo, tooltip

try:
    from aqt.utils import openFolder as _open_folder
except ImportError:  # pragma: no cover - older Anki
    _open_folder = None  # type: ignore[assignment]

from .cleaner import gemini_log as glog
from .cleaner.gemini import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GRAMMAR_INSTRUCTIONS,
    DEFAULT_REQUEST_GAP_SEC,
    DEFAULT_SYSTEM_INSTRUCTIONS,
    DEFAULT_TRANSLATION_INSTRUCTIONS,
    FALLBACK_GEMINI_MODELS,
    GeminiError,
    generate_for_sentence,
    list_gemini_models,
    normalize_model_id,
)

from . import ADDON_PACKAGE, service

_SENTENCE_CANDIDATES = (
    "Sentence",
    "Sentence Furigana",
    "Example",
    "Example Sentence",
    "Expression Sentence",
    "Context",
    "Japanese",
)
_TRANSLATION_CANDIDATES = (
    "Sentence Meaning",
    "Sentence Translation",
    "Translation",
    "English",
    "Sentence EN",
    "EN Translation",
    "Meaning",
)
_GRAMMAR_CANDIDATES = (
    "Grammar Explanation",
    "Grammar",
    "Grammar Note",
    "Notes",
    "Explanation",
)

_SOFT_BATCH_WARN = 40
_HTTP_TIMEOUT_SEC = 45.0
_LOG_FILE = Path(__file__).resolve().parent / "user_files" / "gemini.log"
_STEP_LABELS = ("1 · Setup", "2 · Find notes", "3 · Generate")


def _fmt_duration(seconds: float) -> str:
    """Short clock-style duration for the progress panel."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {rem:02d}s"
    hours, rem_m = divmod(mins, 60)
    return f"{hours}h {rem_m:02d}m"


def _cfg() -> dict:
    return mw.addonManager.getConfig(ADDON_PACKAGE) or {}


def _save_cfg(updates: dict) -> None:
    conf = mw.addonManager.getConfig(ADDON_PACKAGE) or {}
    conf.update(updates)
    mw.addonManager.writeConfig(ADDON_PACKAGE, conf)


def _combo_value(combo: QComboBox) -> str:
    data = combo.currentData()
    if data is None:
        return combo.currentText().strip()
    return str(data or "").strip()


def _pick_default(fields: list[str], preferred: str, candidates: tuple[str, ...]) -> str:
    if preferred and preferred in fields:
        return preferred
    for name in candidates:
        if name in fields:
            return name
    return ""


def _fill_field_combo(combo: QComboBox, fields: list[str], selected: str) -> None:
    combo.blockSignals(True)
    combo.clear()
    for name in fields:
        combo.addItem(name, name)
    idx = combo.findData(selected)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    elif selected:
        combo.addItem(f"{selected} (not on deck notes)", selected)
        combo.setCurrentIndex(combo.count() - 1)
    elif combo.count():
        combo.setCurrentIndex(0)
    combo.blockSignals(False)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class StepIndicator(QWidget):
    """Horizontal step pills near the top of the dialog."""

    def __init__(
        self, labels: tuple[str, ...] = _STEP_LABELS, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pills: list[QLabel] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for i, text in enumerate(labels):
            pill = QLabel(text)
            pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pill.setMinimumHeight(28)
            pill.setStyleSheet(self._style(False, False))
            self._pills.append(pill)
            lay.addWidget(pill, 1)
            if i < len(labels) - 1:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setMaximumWidth(18)
                lay.addWidget(arrow, 0)
        self.set_step(0)

    @staticmethod
    def _style(active: bool, done: bool) -> str:
        if active:
            return (
                "QLabel { background: #2f6fed; color: white; border-radius: 6px; "
                "padding: 4px 8px; font-weight: 600; }"
            )
        if done:
            return (
                "QLabel { background: #2a3a2a; color: #9dcc9d; border-radius: 6px; "
                "padding: 4px 8px; }"
            )
        return (
            "QLabel { background: #2a2a2a; color: #888; border-radius: 6px; "
            "padding: 4px 8px; }"
        )

    def set_step(self, index: int) -> None:
        for i, pill in enumerate(self._pills):
            pill.setStyleSheet(self._style(active=(i == index), done=(i < index)))


class GeminiDialog(QDialog):
    """Three-step wizard: Setup → Find notes → Generate."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        note_ids: list[int] | None = None,
    ) -> None:
        super().__init__(parent or mw)
        self.setWindowTitle("Gemini Sentence Assist")
        self.resize(1000, 760)
        self.setMinimumSize(820, 620)

        conf = _cfg()
        self._step = 0
        self._rows_by_id: dict[int, dict] = {}
        self._known_models: list[str] = list(FALLBACK_GEMINI_MODELS)
        self._busy = False
        self._cancel = False
        self._run_jobs: list = []
        self._run_index = 0
        self._run_results: list = []
        self._run_settings: dict = {}
        self._run_undo = None
        self._run_started = 0.0
        self._card_started = 0.0
        self._run_was_test = False
        self._keep_note_id: int | None = None
        self._pending_preview: service.GeminiFillResult | None = None
        self._pending_job: service.GeminiJob | None = None
        self._pending_browser_ids: list[int] | None = (
            [int(n) for n in note_ids] if note_ids else None
        )
        self._wait_timer = QTimer(self)
        qconnect(self._wait_timer.timeout, self._tick_wait)

        glog.set_log_path(_LOG_FILE)
        glog.log("dialog opened", path=str(_LOG_FILE))

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        title = QLabel(
            "<b>Gemini Sentence Assist</b> — fill English translation and/or "
            "grammar notes from a Japanese sentence field."
        )
        title.setWordWrap(True)
        title.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(title)

        self.steps = StepIndicator()
        root.addWidget(self.steps)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self._build_setup_page(conf)
        self._build_find_page()
        self._build_generate_page()

        footer = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_next = QPushButton("Next")
        self.btn_close = QPushButton("Close")
        self.btn_open_log = QPushButton("Open log")
        self.btn_open_log.setToolTip(f"Open folder containing {_LOG_FILE.name}")
        footer.addWidget(self.btn_back)
        footer.addWidget(self.btn_open_log)
        footer.addStretch(1)
        footer.addWidget(self.btn_close)
        footer.addWidget(self.btn_next)
        root.addLayout(footer)

        qconnect(self.deck.currentTextChanged, self._reload_fields)
        qconnect(self.deck.currentTextChanged, self._update_nav)
        qconnect(self.api_key.textChanged, self._update_nav)
        qconnect(self.want_trans.toggled, self._update_nav)
        qconnect(self.want_gram.toggled, self._update_nav)
        qconnect(self.to_field.currentTextChanged, self._refresh_find_missing_en_label)
        qconnect(self.to_field.editTextChanged, self._refresh_find_missing_en_label)
        qconnect(self.btn_refresh_models.clicked, self._refresh_models)
        qconnect(self.btn_reset_prompts.clicked, self._reset_prompts)
        qconnect(self.btn_find_need.clicked, lambda: self._find(need_only=True))
        qconnect(
            self.btn_find_missing_en.clicked,
            lambda: self._find(need_only=True, require_empty="translation"),
        )
        qconnect(self.btn_find_all.clicked, lambda: self._find(need_only=False))
        qconnect(self.btn_test_one.clicked, lambda: self._run(preview_only=True))
        qconnect(self.btn_apply_preview.clicked, self._apply_pending_preview)
        qconnect(self.btn_run.clicked, lambda: self._run(preview_only=False))
        qconnect(self.btn_cancel.clicked, self._cancel_run)
        qconnect(self.btn_open_log.clicked, self._open_log)
        qconnect(self.pick_list.currentItemChanged, self._on_item_changed)
        qconnect(self.pick_list.itemClicked, self._on_item_clicked)
        qconnect(self.pick_list.itemChanged, self._on_pick_item_changed)
        qconnect(self.btn_select_all.clicked, lambda: self._set_pick_checks(True))
        qconnect(self.btn_clear_all.clicked, lambda: self._set_pick_checks(False))
        qconnect(self.btn_back.clicked, self._go_back)
        qconnect(self.btn_next.clicked, self._go_next)
        qconnect(self.btn_close.clicked, self.reject)

        saved_models = conf.get("gemini_models") or []
        if isinstance(saved_models, list) and saved_models:
            self._known_models = [str(m) for m in saved_models if str(m).strip()]
        self._populate_models(
            normalize_model_id(
                str(conf.get("gemini_model") or DEFAULT_GEMINI_MODEL),
                self._known_models,
            )
        )
        self._refresh_find_missing_en_label()

        self._goto_step(0)
        if self._pending_browser_ids:
            self.load_note_ids(self._pending_browser_ids)

    # ------------------------------------------------------------------ pages
    def _build_setup_page(self, conf: dict) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        hint = QLabel(
            "Connect Gemini and map <b>From (JP)</b> → <b>To (EN)</b>. "
            "Nothing is written until the Generate step."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(12)

        basics = QGroupBox("Connection")
        form = QFormLayout(basics)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setContentsMargins(12, 14, 12, 12)

        self.deck = QComboBox()
        self.deck.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for name in service.list_decks():
            self.deck.addItem(name)
        saved_deck = conf.get("deck", "")
        if saved_deck:
            idx = self.deck.findText(saved_deck)
            if idx >= 0:
                self.deck.setCurrentIndex(idx)

        self.api_key = QLineEdit(conf.get("gemini_api_key", ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Paste key from aistudio.google.com/apikey")

        model_row = QWidget()
        model_lay = QHBoxLayout(model_row)
        model_lay.setContentsMargins(0, 0, 0, 0)
        model_lay.setSpacing(8)
        self.model = QComboBox()
        self.model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_refresh_models = QPushButton("Refresh models")
        self.btn_refresh_models.setToolTip(
            "Call Google’s models.list for this API key and update the dropdown."
        )
        model_lay.addWidget(self.model, 1)
        model_lay.addWidget(self.btn_refresh_models, 0)

        form.addRow("Deck", self.deck)
        form.addRow("API key", self.api_key)
        form.addRow("Model", model_row)
        inner_lay.addWidget(basics)

        mapping = QGroupBox("Fields")
        map_form = QFormLayout(mapping)
        map_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        map_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        map_form.setHorizontalSpacing(12)
        map_form.setVerticalSpacing(10)
        map_form.setContentsMargins(12, 14, 12, 12)

        self.field_names = service.list_deck_note_field_names(self.deck.currentText())
        from_default = _pick_default(
            self.field_names,
            str(conf.get("gemini_sentence_field") or ""),
            _SENTENCE_CANDIDATES,
        )
        to_default = _pick_default(
            self.field_names,
            str(conf.get("gemini_translation_field") or ""),
            _TRANSLATION_CANDIDATES,
        )
        gram_default = _pick_default(
            self.field_names,
            str(conf.get("gemini_grammar_field") or ""),
            _GRAMMAR_CANDIDATES,
        )

        self.from_field = QComboBox()
        self.from_field.setEditable(True)
        self.from_field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        _fill_field_combo(self.from_field, self.field_names, from_default)

        self.want_trans = QCheckBox("Write English translation into To field")
        self.want_trans.setChecked(bool(conf.get("gemini_want_translation", True)))
        if not to_default:
            self.want_trans.setChecked(False)

        self.to_field = QComboBox()
        self.to_field.setEditable(True)
        self.to_field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        _fill_field_combo(self.to_field, self.field_names, to_default)

        self.want_gram = QCheckBox("Also write short grammar explanation")
        want_gram = bool(conf.get("gemini_want_grammar", False)) and bool(gram_default)
        self.want_gram.setChecked(want_gram)

        self.grammar_field = QComboBox()
        self.grammar_field.setEditable(True)
        self.grammar_field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        _fill_field_combo(self.grammar_field, self.field_names, gram_default)

        self.overwrite = QCheckBox("Overwrite existing To / Grammar text")
        self.overwrite.setChecked(bool(conf.get("gemini_overwrite", False)))

        map_form.addRow("From (JP)", self.from_field)
        map_form.addRow("", self.want_trans)
        map_form.addRow("To (EN)", self.to_field)
        map_form.addRow("", self.want_gram)
        map_form.addRow("Grammar", self.grammar_field)
        map_form.addRow("", self.overwrite)

        tip = QLabel(
            "Dropdowns only show fields found on notes in this deck. "
            "If Find returns 0, change To to a real field (e.g. Sentence Meaning)."
        )
        tip.setWordWrap(True)
        map_form.addRow(tip)
        inner_lay.addWidget(mapping)

        prompts = QGroupBox("Prompts (optional)")
        p_lay = QVBoxLayout(prompts)
        p_lay.setContentsMargins(12, 14, 12, 12)
        p_lay.setSpacing(8)

        p_tip = QLabel(
            "Defaults forbid romaji. Edit if you want different tone or detail. "
            "Saved automatically when you continue."
        )
        p_tip.setWordWrap(True)
        p_lay.addWidget(p_tip)

        self.prompt_system = QTextEdit()
        self.prompt_system.setAcceptRichText(False)
        self.prompt_system.setPlainText(
            str(conf.get("gemini_system_instructions") or DEFAULT_SYSTEM_INSTRUCTIONS)
        )
        self.prompt_system.setMaximumHeight(80)

        self.prompt_translation = QTextEdit()
        self.prompt_translation.setAcceptRichText(False)
        self.prompt_translation.setPlainText(
            str(
                conf.get("gemini_translation_instructions")
                or DEFAULT_TRANSLATION_INSTRUCTIONS
            )
        )
        self.prompt_translation.setMaximumHeight(64)

        self.prompt_grammar = QTextEdit()
        self.prompt_grammar.setAcceptRichText(False)
        self.prompt_grammar.setPlainText(
            str(conf.get("gemini_grammar_instructions") or DEFAULT_GRAMMAR_INSTRUCTIONS)
        )
        self.prompt_grammar.setMaximumHeight(80)

        p_lay.addWidget(QLabel("System / overall"))
        p_lay.addWidget(self.prompt_system)
        p_lay.addWidget(QLabel("Translation instructions"))
        p_lay.addWidget(self.prompt_translation)
        p_lay.addWidget(QLabel("Grammar instructions"))
        p_lay.addWidget(self.prompt_grammar)

        reset_row = QHBoxLayout()
        self.btn_reset_prompts = QPushButton("Reset prompts to defaults")
        reset_row.addWidget(self.btn_reset_prompts)
        reset_row.addStretch(1)
        p_lay.addLayout(reset_row)
        inner_lay.addWidget(prompts)
        inner_lay.addStretch(1)

        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        self.stack.addWidget(page)

    def _build_find_page(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        hint = QLabel(
            "Find notes that have a Japanese sentence, then select the ones to fill. "
            "Click a row to inspect current EN / grammar."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        actions = QHBoxLayout()
        self.btn_find_need = QPushButton("Find needing fill")
        self.btn_find_missing_en = QPushButton('Find without “Sentence Meaning”')
        self.btn_find_all = QPushButton("List all with JP sentence")
        self.btn_find_need.setToolTip(
            "Notes that still need EN and/or grammar (respects Overwrite)."
        )
        self.btn_find_missing_en.setToolTip(
            "Notes with a JP sentence whose To (EN) field is empty — "
            "e.g. empty Sentence Meaning. Ignores Overwrite."
        )
        self.btn_find_all.setToolTip("All notes in the deck with a non-empty From field.")
        actions.addWidget(self.btn_find_need)
        actions.addWidget(self.btn_find_missing_en)
        actions.addWidget(self.btn_find_all)
        actions.addStretch(1)
        lay.addLayout(actions)

        self.find_meta = QLabel(
            "Find needing fill, Find without To (EN), or open from Browser selection."
        )
        self.find_meta.setWordWrap(True)
        self.find_meta.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.find_meta)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("Found notes"))
        check_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select all")
        self.btn_clear_all = QPushButton("Clear all")
        check_row.addWidget(self.btn_select_all)
        check_row.addWidget(self.btn_clear_all)
        check_row.addStretch(1)
        left_lay.addLayout(check_row)
        self.pick_list = QListWidget()
        self.pick_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.pick_list.setAlternatingRowColors(True)
        left_lay.addWidget(self.pick_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("Current fields"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setAcceptRichText(True)
        self.preview.setPlaceholderText("Select a found note to preview JP / EN / grammar.")
        right_lay.addWidget(self.preview, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 480])
        lay.addWidget(splitter, 1)

        self.stack.addWidget(page)

    def _build_generate_page(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        hint = QLabel(
            "Preview one card first (safe — does not write), then Apply, "
            "or Fill selected to write immediately."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.gen_summary = QLabel("<b>0 notes</b> selected")
        self.gen_summary.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.gen_summary)

        actions = QHBoxLayout()
        self.btn_test_one = QPushButton("Preview selected (1)")
        self.btn_apply_preview = QPushButton("Apply preview")
        self.btn_run = QPushButton("Fill selected with Gemini")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_apply_preview.setEnabled(False)
        self.btn_test_one.setToolTip(
            "Call Gemini for one selected note and show the result — does NOT write the card."
        )
        self.btn_apply_preview.setToolTip(
            "Write the last Preview result to the card (no second API call)."
        )
        self.btn_run.setToolTip(
            "Call Gemini for selected notes and write EN / grammar fields immediately."
        )
        actions.addWidget(self.btn_test_one)
        actions.addWidget(self.btn_apply_preview)
        actions.addWidget(self.btn_run)
        actions.addWidget(self.btn_cancel)
        actions.addStretch(1)
        lay.addLayout(actions)

        self.status = QLabel("Ready — Anki stays usable while Gemini runs in the background.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.progress_panel = QFrame()
        self.progress_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.progress_panel.setVisible(False)
        prog_lay = QVBoxLayout(self.progress_panel)
        prog_lay.setContentsMargins(10, 8, 10, 8)
        prog_lay.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.progress_title = QLabel("Working…")
        self.progress_title.setTextFormat(Qt.TextFormat.RichText)
        self.progress_pct = QLabel("")
        self.progress_pct.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        head.addWidget(self.progress_title, 1)
        head.addWidget(self.progress_pct, 0)
        prog_lay.addLayout(head)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m cards")
        self.progress_bar.setMinimumHeight(22)
        prog_lay.addWidget(self.progress_bar)

        self.progress_detail = QLabel("")
        self.progress_detail.setWordWrap(True)
        self.progress_detail.setTextFormat(Qt.TextFormat.RichText)
        prog_lay.addWidget(self.progress_detail)
        lay.addWidget(self.progress_panel)

        self.meta = QLabel(
            f"Log: <code>{_escape(str(_LOG_FILE))}</code>"
        )
        self.meta.setWordWrap(True)
        self.meta.setTextFormat(Qt.TextFormat.RichText)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.meta)

        lay.addWidget(QLabel("Result preview"))
        self.gen_preview = QTextEdit()
        self.gen_preview.setReadOnly(True)
        self.gen_preview.setAcceptRichText(True)
        self.gen_preview.setPlaceholderText(
            "Preview selected (1) shows Gemini output here before you Apply."
        )
        lay.addWidget(self.gen_preview, 1)

        self.stack.addWidget(page)

    # ------------------------------------------------------------------ nav
    def _goto_step(self, index: int) -> None:
        self._step = max(0, min(2, index))
        if self._step == 2:
            self._refresh_generate_summary()
            # Mirror current note into generate preview if no pending Gemini preview
            if self._pending_preview is None:
                self._mirror_find_preview_to_generate()
        self.stack.setCurrentIndex(self._step)
        self.steps.set_step(self._step)
        self._update_nav()

    def _go_back(self) -> None:
        if self._busy:
            showInfo("Wait for Gemini to finish (or Cancel) before leaving this step.")
            return
        if self._step > 0:
            self._goto_step(self._step - 1)

    def _go_next(self) -> None:
        if self._busy:
            showInfo("Wait for Gemini to finish (or Cancel) before continuing.")
            return
        if self._step == 0:
            if not self._setup_ready():
                showInfo(
                    "Paste an API key and enable English and/or grammar before continuing."
                )
                return
            self._persist()
            self._goto_step(1)
            return
        if self._step == 1:
            if self.pick_list.count() < 1:
                showInfo("Find notes first (or open from Browser selection).")
                return
            if not self._selected_ids():
                showInfo("Tick one or more notes (or Select all), then Next.")
                return
            self._goto_step(2)

    def _update_nav(self) -> None:
        self.btn_back.setEnabled(self._step > 0 and not self._busy)
        on_last = self._step == 2
        self.btn_next.setVisible(not on_last)
        if self._step == 0:
            self.btn_next.setEnabled(self._setup_ready() and not self._busy)
            self.btn_next.setDefault(True)
        elif self._step == 1:
            self.btn_next.setEnabled(bool(self._selected_ids()) and not self._busy)
            self.btn_next.setDefault(True)

    def _setup_ready(self) -> bool:
        if not self.api_key.text().strip():
            return False
        if not self.want_trans.isChecked() and not self.want_gram.isChecked():
            return False
        return True

    def _on_pick_item_changed(self, _item: QListWidgetItem) -> None:
        if self._step == 2:
            self._refresh_generate_summary()
        self._update_nav()

    def _set_pick_checks(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.pick_list.blockSignals(True)
        for i in range(self.pick_list.count()):
            item = self.pick_list.item(i)
            if item is not None:
                item.setCheckState(state)
        self.pick_list.blockSignals(False)
        self._on_pick_item_changed(None)  # type: ignore[arg-type]

    def _refresh_generate_summary(self) -> None:
        ids = self._selected_ids()
        sel = len(ids)
        listed = self.pick_list.count()
        if sel:
            self.gen_summary.setText(
                f"<b>{sel} note(s)</b> ticked for Generate"
                + (f" · {listed} listed" if listed != sel else "")
            )
        else:
            self.gen_summary.setText(
                f"<b>0 note(s)</b> ticked — go Back and tick rows, or Select all"
            )
        self.gen_summary.setTextFormat(Qt.TextFormat.RichText)

    def _mirror_find_preview_to_generate(self) -> None:
        """Copy the Find-step HTML preview into Generate when idle."""
        html = self.preview.toHtml().strip()
        if html:
            self.gen_preview.setHtml(html)

    # ------------------------------------------------------------------ models / config
    def _populate_models(self, selected: str) -> None:
        models = self._known_models or list(FALLBACK_GEMINI_MODELS)
        self.model.blockSignals(True)
        self.model.clear()
        for mid in models:
            self.model.addItem(mid, mid)
        selected = normalize_model_id(selected, models)
        idx = self.model.findData(selected)
        if idx >= 0:
            self.model.setCurrentIndex(idx)
        elif self.model.count():
            self.model.setCurrentIndex(0)
        self.model.blockSignals(False)

    def _refresh_models(self) -> None:
        key = self.api_key.text().strip()
        if not key:
            showInfo("Paste your Gemini API key first, then Refresh models.")
            return
        try:
            models = list_gemini_models(key)
        except GeminiError as exc:
            QMessageBox.warning(self, "Refresh models failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Refresh models failed", str(exc))
            return

        current = _combo_value(self.model)
        self._known_models = models
        self._populate_models(current)
        _save_cfg({"gemini_models": models, "gemini_model": _combo_value(self.model)})
        tooltip(f"Loaded {len(models)} model(s) from Google.")

    def _reload_fields(self, _deck: str = "") -> None:
        self.field_names = service.list_deck_note_field_names(self.deck.currentText())
        _fill_field_combo(self.from_field, self.field_names, _combo_value(self.from_field))
        _fill_field_combo(self.to_field, self.field_names, _combo_value(self.to_field))
        _fill_field_combo(
            self.grammar_field, self.field_names, _combo_value(self.grammar_field)
        )
        self._refresh_find_missing_en_label()

    def _reset_prompts(self) -> None:
        self.prompt_system.setPlainText(DEFAULT_SYSTEM_INSTRUCTIONS)
        self.prompt_translation.setPlainText(DEFAULT_TRANSLATION_INSTRUCTIONS)
        self.prompt_grammar.setPlainText(DEFAULT_GRAMMAR_INSTRUCTIONS)
        self._persist()
        tooltip("Prompts reset (includes no-romaji rules).")

    def _persist(self) -> None:
        _save_cfg(
            {
                "deck": self.deck.currentText(),
                "gemini_api_key": self.api_key.text().strip(),
                "gemini_model": _combo_value(self.model) or DEFAULT_GEMINI_MODEL,
                "gemini_models": self._known_models,
                "gemini_sentence_field": _combo_value(self.from_field),
                "gemini_translation_field": _combo_value(self.to_field),
                "gemini_grammar_field": _combo_value(self.grammar_field),
                "gemini_want_translation": self.want_trans.isChecked(),
                "gemini_want_grammar": self.want_gram.isChecked(),
                "gemini_overwrite": self.overwrite.isChecked(),
                "gemini_system_instructions": self.prompt_system.toPlainText().strip(),
                "gemini_translation_instructions": self.prompt_translation.toPlainText().strip(),
                "gemini_grammar_instructions": self.prompt_grammar.toPlainText().strip(),
            }
        )

    def _settings(self) -> dict:
        return {
            "api_key": self.api_key.text().strip(),
            "model": normalize_model_id(
                _combo_value(self.model), self._known_models or None
            ),
            "sentence_field": _combo_value(self.from_field),
            "translation_field": _combo_value(self.to_field),
            "grammar_field": _combo_value(self.grammar_field),
            "want_translation": self.want_trans.isChecked(),
            "want_grammar": self.want_gram.isChecked(),
            "overwrite": self.overwrite.isChecked(),
            "system_instructions": self.prompt_system.toPlainText().strip()
            or DEFAULT_SYSTEM_INSTRUCTIONS,
            "translation_instructions": self.prompt_translation.toPlainText().strip()
            or DEFAULT_TRANSLATION_INSTRUCTIONS,
            "grammar_instructions": self.prompt_grammar.toPlainText().strip()
            or DEFAULT_GRAMMAR_INSTRUCTIONS,
        }

    # ------------------------------------------------------------------ find / preview
    def _refresh_find_missing_en_label(self, *_args) -> None:
        name = _combo_value(self.to_field) or "Sentence Meaning"
        self.btn_find_missing_en.setText(f'Find without “{name}”')

    def _find(self, *, need_only: bool, require_empty: str | None = None) -> None:
        self._persist()
        s = self._settings()
        if require_empty == "translation" and not s["translation_field"]:
            showInfo("Pick a To (EN) field on Setup first.")
            return
        if not s["want_translation"] and not s["want_grammar"] and require_empty is None:
            showInfo("Enable English translation and/or grammar.")
            return
        try:
            result = service.list_gemini_candidates(
                self.deck.currentText(),
                s["sentence_field"],
                s["translation_field"],
                s["grammar_field"],
                want_translation=s["want_translation"],
                want_grammar=s["want_grammar"],
                overwrite=s["overwrite"],
                need_only=need_only,
                require_empty=require_empty,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Find failed", str(exc))
            return

        self._populate_list_from_rows(result.rows, select_all=False)
        summary = result.summary_html()
        if require_empty == "translation":
            summary = (
                f"<b>Empty To ({_escape(result.to_field)})</b> — "
                f"{len(result.rows)} note(s) with JP sentence and blank EN.<br>"
                + summary
            )
        self.find_meta.setText(summary)
        self.find_meta.setTextFormat(Qt.TextFormat.RichText)
        if result.rows:
            self.pick_list.setCurrentRow(0)
        self._update_nav()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._show_preview(item)

    def _on_item_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        self._show_preview(current)

    def _show_preview(self, item: QListWidgetItem | None) -> None:
        if item is None:
            self.preview.clear()
            return
        note_id = int(item.data(Qt.ItemDataRole.UserRole))
        row = self._rows_by_id.get(note_id)
        if not row:
            self.preview.setPlainText(f"Note {note_id}")
            return
        flags = []
        if row.get("need_translation"):
            flags.append("needs EN")
        if row.get("need_grammar"):
            flags.append("needs grammar")
        status = ", ".join(flags) if flags else "targets already filled"
        en = row.get("translation_preview") or "(empty)"
        gram = row.get("grammar_preview") or "(empty)"
        html = (
            f"<p><b>{_escape(row.get('expression') or '')}</b> · note {note_id}<br>"
            f"<i>{_escape(status)}</i></p>"
            f"<p><b>From (JP)</b><br>{_escape(row.get('preview') or '')}</p>"
            f"<p><b>To (EN) now</b><br>{_escape(en)}</p>"
            f"<p><b>Grammar now</b><br>{_escape(gram)}</p>"
        )
        self.preview.setHtml(html)
        if self._step == 2 and self._pending_preview is None:
            self.gen_preview.setHtml(html)

    def _selected_ids(self) -> list[int]:
        """Note IDs with a checked tick box."""
        ids: list[int] = []
        for i in range(self.pick_list.count()):
            item = self.pick_list.item(i)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    # ------------------------------------------------------------------ progress / run
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.btn_find_need.setEnabled(not busy)
        self.btn_find_missing_en.setEnabled(not busy)
        self.btn_find_all.setEnabled(not busy)
        self.btn_select_all.setEnabled(not busy)
        self.btn_clear_all.setEnabled(not busy)
        self.btn_test_one.setEnabled(not busy)
        self.btn_run.setEnabled(not busy)
        self.btn_refresh_models.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.progress_panel.setVisible(busy)
        self._update_nav()
        if not busy:
            self.btn_apply_preview.setEnabled(self._pending_preview is not None)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(100)
            self.progress_title.setText("Working…")
            self.progress_pct.setText("")
            self.progress_detail.setText("")
        else:
            self.btn_apply_preview.setEnabled(False)

    def _progress_counts(self) -> tuple[int, int, int]:
        """Return (done, current_1based_or_0, total)."""
        total = len(self._run_jobs) if self._run_jobs else 0
        done = len(self._run_results)
        if not total:
            return 0, 0, 0
        if done >= total:
            return total, total, total
        return done, done + 1, total

    def _update_progress(self, detail: str, *, phase: str = "working") -> None:
        """Refresh the visual progress panel during a run."""
        done, current, total = self._progress_counts()
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(min(done, total))

        if total <= 0:
            self.progress_title.setText("<b>Working…</b>")
            self.progress_pct.setText("")
        elif phase == "done":
            self.progress_title.setText(f"<b>Finished</b> — {total} card(s)")
            self.progress_pct.setText("100%")
            self.progress_bar.setValue(total)
        elif phase == "cancel":
            self.progress_title.setText(
                f"<b>Cancelling…</b> — card {current} of {total}"
            )
            pct = int(round(100 * done / total)) if total else 0
            self.progress_pct.setText(f"{pct}%")
        elif phase == "pacing":
            self.progress_title.setText(
                f"<b>{done} / {total}</b> done — pacing before next"
            )
            pct = int(round(100 * done / total)) if total else 0
            self.progress_pct.setText(f"{pct}%")
        else:
            self.progress_title.setText(
                f"<b>Card {current} of {total}</b>"
                + (f" · {done} done" if done else "")
            )
            pct = int(round(100 * done / total)) if total else 0
            self.progress_pct.setText(f"{pct}%")

        timings = [
            r.elapsed_s
            for r in self._run_results
            if r.elapsed_s is not None and r.elapsed_s > 0
        ]
        bits: list[str] = []
        if detail:
            bits.append(detail)
        if self._busy and getattr(self, "_run_started", 0):
            total_elapsed = time.monotonic() - self._run_started
            bits.append(f"elapsed {_fmt_duration(total_elapsed)}")
            remaining = max(0, total - done)
            if timings and remaining > 0 and phase not in {"done", "cancel"}:
                avg = sum(timings) / len(timings)
                gap = DEFAULT_REQUEST_GAP_SEC if total > 1 else 0.0
                eta = remaining * avg + max(0, remaining - 1) * gap
                bits.append(f"~{_fmt_duration(eta)} left")
            elif phase == "working" and getattr(self, "_card_started", 0):
                card_s = time.monotonic() - self._card_started
                if card_s >= 1:
                    bits.append(f"this request {_fmt_duration(card_s)}")
        self.progress_detail.setText(" · ".join(bits))
        if phase == "done":
            return
        if total > 0:
            pct = int(round(100 * done / total))
            self.status.setText(
                f"{'Cancelling' if phase == 'cancel' else 'Working'} — "
                f"{done}/{total} cards ({pct}%)"
            )

    def _open_log(self) -> None:
        path = glog.log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("Gemini log\n", encoding="utf-8")
        try:
            if _open_folder is not None:
                _open_folder(str(path.parent))
            else:
                showInfo(f"Log file:\n{path}")
        except Exception:  # noqa: BLE001
            showInfo(f"Log file:\n{path}")

    def _cancel_run(self) -> None:
        if not self._busy:
            return
        self._cancel = True
        glog.log("cancel requested by user")
        self._update_progress(
            "waiting for current HTTP call to finish…",
            phase="cancel",
        )

    def _tick_wait(self) -> None:
        if not self._busy:
            self._wait_timer.stop()
            return
        if not self._run_jobs or self._run_index >= len(self._run_jobs):
            return
        job = self._run_jobs[self._run_index]
        model = self._run_settings.get("model", "?")
        expr = (job.expression or "")[:32]
        label = f"<b>{_escape(expr)}</b>" if expr else f"note {job.note_id}"
        self._update_progress(
            f"Calling <code>{_escape(str(model))}</code> for {label} — waiting for network",
            phase="working",
        )

    def _run(self, *, preview_only: bool) -> None:
        if self._busy:
            showInfo("Gemini is already running — Cancel first or wait.")
            return
        if self._step != 2:
            self._goto_step(2)
        self._persist()
        s = self._settings()
        if not s["api_key"]:
            showInfo("Paste your Gemini API key first (aistudio.google.com/apikey).")
            return
        if not s["want_translation"] and not s["want_grammar"]:
            showInfo("Enable English translation and/or grammar.")
            return

        ids = self._selected_ids()
        if preview_only:
            # Prefer ticked; else the highlighted row for a quick one-off preview
            if not ids:
                cur = self.pick_list.currentItem()
                if cur is not None:
                    ids = [int(cur.data(Qt.ItemDataRole.UserRole))]
            if not ids:
                showInfo("Tick or highlight one note first.")
                return
            ids = ids[:1]
        elif not ids and self.pick_list.count():
            reply = QMessageBox.question(
                self,
                "Gemini fill",
                f"No notes ticked. Fill all {self.pick_list.count()} listed notes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._set_pick_checks(True)
                ids = self._selected_ids()
        if not ids:
            showInfo("Tick one or more notes on Find (Select all), then Generate.")
            return

        if not preview_only and len(ids) > _SOFT_BATCH_WARN:
            mins = max(1, int(round(len(ids) * DEFAULT_REQUEST_GAP_SEC / 60)))
            reply = QMessageBox.question(
                self,
                "Large Gemini batch",
                f"You’re about to call Gemini and WRITE {len(ids)} notes "
                f"(~{mins} min on free-tier pacing).\n\n"
                "Tip: Preview selected (1) first.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        elif not preview_only:
            mins = max(1, int(round(len(ids) * DEFAULT_REQUEST_GAP_SEC / 60)))
            reply = QMessageBox.question(
                self,
                "Gemini fill",
                f"Call Gemini ({s['model']}) and WRITE {len(ids)} note(s)?\n"
                f"Runs in the background — Anki should stay usable.\n"
                f"About {mins} minute(s) on free-tier pacing.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        jobs = service.collect_gemini_jobs(
            ids,
            sentence_field=s["sentence_field"],
            translation_field=s["translation_field"],
            grammar_field=s["grammar_field"],
            want_translation=s["want_translation"],
            want_grammar=s["want_grammar"],
            overwrite=True if preview_only else s["overwrite"],
        )
        if not jobs:
            showInfo(
                "Nothing to do for the selection "
                "(targets already filled, or fields missing). Try Overwrite."
            )
            return

        glog.log(
            "run start",
            model=s["model"],
            notes=len(jobs),
            preview_only=preview_only,
            timeout_s=_HTTP_TIMEOUT_SEC,
        )
        assert mw.col is not None
        self._run_jobs = jobs
        self._run_index = 0
        self._run_results = []
        self._run_settings = s
        self._run_undo = (
            None
            if preview_only
            else mw.col.add_custom_undo_entry("Gemini Sentence Assist")
        )
        self._run_was_test = bool(preview_only)
        self._keep_note_id = jobs[0].note_id if jobs else None
        if preview_only:
            self._pending_preview = None
            self._pending_job = None
            self.btn_apply_preview.setEnabled(False)
        self._cancel = False
        self._run_started = time.monotonic()
        self._set_busy(True)
        self._wait_timer.start(500)
        mode = "preview only (no write)" if preview_only else "writing results"
        self._update_progress(
            f"Starting <code>{_escape(s['model'])}</code> — {mode}",
            phase="working",
        )
        self._start_next_job()

    def _start_next_job(self) -> None:
        if self._cancel:
            glog.log("run cancelled", done=self._run_index, total=len(self._run_jobs))
            self._finish_run(cancelled=True)
            return
        if self._run_index >= len(self._run_jobs):
            self._finish_run(cancelled=False)
            return

        job = self._run_jobs[self._run_index]
        s = self._run_settings
        self._card_started = time.monotonic()
        glog.log(
            "card start",
            note_id=job.note_id,
            index=self._run_index + 1,
            total=len(self._run_jobs),
            expression=job.expression[:40],
            sentence_chars=len(job.sentence),
            do_trans=job.do_trans,
            do_gram=job.do_gram,
            model=s["model"],
        )
        expr = (job.expression or "")[:40]
        label = f"<b>{_escape(expr)}</b>" if expr else f"note {job.note_id}"
        self._update_progress(
            f"Calling <code>{_escape(s['model'])}</code> for {label} — waiting for network",
            phase="working",
        )

        api_key = s["api_key"]
        model = s["model"]
        sentence = job.sentence
        expression = job.expression
        do_trans = job.do_trans
        do_gram = job.do_gram
        system_instructions = s["system_instructions"]
        translation_instructions = s["translation_instructions"]
        grammar_instructions = s["grammar_instructions"]

        def _bg_call():
            with glog.timed(
                "generate_for_sentence",
                note_id=job.note_id,
                model=model,
            ):
                return generate_for_sentence(
                    api_key,
                    sentence,
                    model=model,
                    want_translation=do_trans,
                    want_grammar=do_gram,
                    expression=expression,
                    timeout_sec=_HTTP_TIMEOUT_SEC,
                    system_instructions=system_instructions,
                    translation_instructions=translation_instructions,
                    grammar_instructions=grammar_instructions,
                )

        def _on_done(fut) -> None:
            self._on_job_done(job, fut)

        if hasattr(mw, "taskman") and hasattr(mw.taskman, "run_in_background"):
            mw.taskman.run_in_background(_bg_call, _on_done)
        else:
            glog.log("WARNING no taskman; falling back to blocking call")
            try:
                class _F:
                    def __init__(self, val=None, err=None):
                        self._val = val
                        self._err = err

                    def result(self):
                        if self._err:
                            raise self._err
                        return self._val

                gem = _bg_call()
                _on_done(_F(gem))
            except Exception as exc:  # noqa: BLE001
                _on_done(_F(err=exc))

    def _on_job_done(self, job, fut) -> None:
        elapsed = time.monotonic() - self._card_started
        try:
            gem = fut.result()
        except Exception as exc:  # noqa: BLE001
            glog.log(
                "card fail",
                note_id=job.note_id,
                elapsed_s=f"{elapsed:.2f}",
                error=str(exc),
            )
            self._run_results.append(
                service.GeminiFillResult(
                    note_id=job.note_id,
                    ok=False,
                    error=str(exc),
                    elapsed_s=elapsed,
                    expression=job.expression,
                    sentence=job.sentence,
                )
            )
            err_l = str(exc).lower()
            if "api key" in err_l or "403" in err_l or "401" in err_l or "404" in err_l:
                self._cancel = True
            self._run_index += 1
            QTimer.singleShot(0, self._start_next_job)
            return

        try:
            result = service.apply_gemini_result(
                job,
                gem,
                translation_field=self._run_settings["translation_field"],
                grammar_field=self._run_settings["grammar_field"],
                undo=self._run_undo,
                elapsed_s=elapsed,
                write=not self._run_was_test,
            )
            self._run_results.append(result)
            if self._run_was_test and result.ok:
                self._pending_preview = result
                self._pending_job = job
            glog.log(
                "card ok",
                note_id=job.note_id,
                elapsed_s=f"{elapsed:.2f}",
                wrote_t=result.wrote_translation,
                wrote_g=result.wrote_grammar,
                preview_only=self._run_was_test,
            )
            self._update_progress(
                f"Finished card in {elapsed:.1f}s"
                + (" — preview ready" if self._run_was_test else " — continuing…"),
                phase="working",
            )
        except Exception as exc:  # noqa: BLE001
            glog.log("card write fail", note_id=job.note_id, error=str(exc))
            self._run_results.append(
                service.GeminiFillResult(
                    note_id=job.note_id,
                    ok=False,
                    error=str(exc),
                    elapsed_s=elapsed,
                    expression=job.expression,
                    sentence=job.sentence,
                )
            )

        self._run_index += 1
        delay_ms = 0
        if (
            not self._cancel
            and self._run_index < len(self._run_jobs)
            and len(self._run_jobs) > 1
        ):
            delay_ms = int(DEFAULT_REQUEST_GAP_SEC * 1000)
            self._update_progress(
                f"Waiting {DEFAULT_REQUEST_GAP_SEC:.0f}s (rate limit pacing)",
                phase="pacing",
            )
        QTimer.singleShot(delay_ms, self._start_next_job)

    def _select_note_in_list(self, note_id: int) -> None:
        for i in range(self.pick_list.count()):
            item = self.pick_list.item(i)
            if item is None:
                continue
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(note_id):
                self.pick_list.setCurrentItem(item)
                self.pick_list.scrollToItem(item)
                self.pick_list.setFocus()
                return

    def _show_test_result(self, result: service.GeminiFillResult) -> None:
        """Keep focus on the tested card and show Gemini output in Preview."""
        note_id = int(result.note_id)
        row = self._rows_by_id.get(note_id, {})
        if result.sentence:
            row = dict(row)
            row["note_id"] = note_id
            row["expression"] = result.expression or row.get("expression") or ""
            row["preview"] = result.sentence
            if result.translation_text:
                row["translation_preview"] = result.translation_text
            if result.grammar_text:
                row["grammar_preview"] = result.grammar_text
            self._rows_by_id[note_id] = row

        self._select_note_in_list(note_id)

        if not result.ok:
            html = (
                f"<p><b>Preview failed</b> · note {note_id}</p>"
                f"<p>{_escape(result.error or 'unknown error')}</p>"
            )
            self.preview.setHtml(html)
            self.gen_preview.setHtml(html)
            return

        elapsed = f"{result.elapsed_s:.1f}s" if result.elapsed_s is not None else "?"
        flags = []
        if result.translation_text:
            flags.append("EN ready")
        if result.grammar_text:
            flags.append("grammar ready")
        if result.skipped:
            flags.append("empty response")
        status = ", ".join(flags) or "done"
        en = result.translation_text or "(empty)"
        gram = result.grammar_text or "(empty)"
        jp = result.sentence or row.get("preview") or ""
        html = (
            f"<p><b>Preview (not written yet)</b> · {_escape(result.expression or '')} · "
            f"note {note_id} · {elapsed}<br>"
            f"<i>{_escape(status)} — click <b>Apply preview</b> to save to the card</i></p>"
            f"<p><b>From (JP)</b><br>{_escape(jp)}</p>"
            f"<p><b>Proposed To (EN)</b><br>{_escape(en)}</p>"
            f"<p><b>Proposed Grammar</b><br>{_escape(gram)}</p>"
        )
        self.preview.setHtml(html)
        self.gen_preview.setHtml(html)

    def _apply_pending_preview(self) -> None:
        if self._busy:
            showInfo("Wait for the current Gemini run to finish.")
            return
        pending = self._pending_preview
        job = self._pending_job
        if pending is None or job is None or not pending.ok:
            showInfo("No preview to apply — run Preview selected (1) first.")
            return
        self._persist()
        s = self._settings()
        assert mw.col is not None
        undo = mw.col.add_custom_undo_entry("Gemini Sentence Assist (apply preview)")

        class _Gem:
            translation = pending.translation_text
            grammar = pending.grammar_text

        result = service.apply_gemini_result(
            job,
            _Gem(),
            translation_field=s["translation_field"],
            grammar_field=s["grammar_field"],
            undo=undo,
            elapsed_s=pending.elapsed_s,
            write=True,
        )
        glog.log(
            "preview applied",
            note_id=result.note_id,
            wrote_t=result.wrote_translation,
            wrote_g=result.wrote_grammar,
        )
        self._pending_preview = None
        self._pending_job = None
        self.btn_apply_preview.setEnabled(False)
        row = self._rows_by_id.get(result.note_id, {})
        row = dict(row)
        row["translation_preview"] = result.translation_text
        row["grammar_preview"] = result.grammar_text
        row["need_translation"] = False
        row["need_grammar"] = False
        self._rows_by_id[result.note_id] = row
        self._select_note_in_list(result.note_id)
        en = result.translation_text or "(empty)"
        gram = result.grammar_text or "(empty)"
        html = (
            f"<p><b>Applied to card</b> · note {result.note_id}</p>"
            f"<p><b>To (EN)</b><br>{_escape(en)}</p>"
            f"<p><b>Grammar</b><br>{_escape(gram)}</p>"
        )
        self.preview.setHtml(html)
        self.gen_preview.setHtml(html)
        msg = f"Applied preview to note {result.note_id}."
        self.status.setText(msg)
        tooltip(msg)

    def _populate_list_from_rows(
        self, rows: list[dict], *, select_all: bool = False
    ) -> None:
        self._rows_by_id = {int(r["note_id"]): r for r in rows}
        self.pick_list.blockSignals(True)
        self.pick_list.clear()
        self.preview.clear()
        self.gen_preview.clear()
        self._pending_preview = None
        self._pending_job = None
        self.btn_apply_preview.setEnabled(False)
        check = Qt.CheckState.Checked if select_all else Qt.CheckState.Unchecked
        for row in rows:
            flags = []
            if row.get("need_translation"):
                flags.append("EN")
            if row.get("need_grammar"):
                flags.append("grammar")
            need = "+".join(flags) if flags else "review"
            snip = (row.get("preview") or "").replace("\n", " ")
            if len(snip) > 60:
                snip = snip[:59] + "…"
            item = QListWidgetItem(f"{row['expression']}  · {need}\n{snip}")
            item.setData(Qt.ItemDataRole.UserRole, int(row["note_id"]))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(check)
            self.pick_list.addItem(item)
        self.pick_list.blockSignals(False)
        if rows:
            self.pick_list.setCurrentRow(0)
        self._update_nav()
        if self._step == 2:
            self._refresh_generate_summary()

    def load_note_ids(self, note_ids: list[int]) -> None:
        """Populate list from Browser (or other) note selection."""
        self._persist()
        s = self._settings()
        ids = [int(n) for n in note_ids]
        rows = service.describe_notes_for_gemini(
            ids,
            sentence_field=s["sentence_field"],
            translation_field=s["translation_field"],
            grammar_field=s["grammar_field"],
            want_translation=s["want_translation"],
            want_grammar=s["want_grammar"],
            overwrite=s["overwrite"],
        )
        self._populate_list_from_rows(rows, select_all=True)
        missing = len(ids) - len(rows)
        self.find_meta.setText(
            f"<b>{len(rows)} note(s)</b> from Browser selection"
            + (f" · skipped {missing} (no From text / field)" if missing else "")
            + "<br>Tick the notes to fill (or <b>Select all</b>), then click <b>Next</b> to Generate."
        )
        self.find_meta.setTextFormat(Qt.TextFormat.RichText)
        self._goto_step(1)
        glog.log("loaded browser notes", requested=len(ids), listed=len(rows))

    def _finish_run(self, *, cancelled: bool) -> None:
        self._wait_timer.stop()
        self._set_busy(False)
        results = self._run_results
        total_elapsed = time.monotonic() - self._run_started
        ok = sum(1 for r in results if r.ok and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        fail = [r for r in results if not r.ok]
        wrote_t = sum(1 for r in results if r.wrote_translation)
        wrote_g = sum(1 for r in results if r.wrote_grammar)
        timings = [r.elapsed_s for r in results if r.elapsed_s is not None]
        avg = (sum(timings) / len(timings)) if timings else 0.0
        prefix = "Cancelled. " if cancelled else ""
        was_preview = bool(self._run_was_test)
        if was_preview:
            msg = (
                f"{prefix}Preview ready for {ok} note(s) "
                f"(empty {skipped}) in {total_elapsed:.1f}s"
                + (f" · {avg:.1f}s" if timings else "")
                + " — not written. Click Apply preview to save."
            )
        else:
            msg = (
                f"{prefix}Updated {ok} note(s) "
                f"(EN×{wrote_t}, grammar×{wrote_g}; skipped {skipped}) "
                f"in {total_elapsed:.1f}s"
                + (f" · avg {avg:.1f}s/card" if timings else "")
            )
        glog.log(
            "run finish",
            cancelled=cancelled,
            preview_only=was_preview,
            ok=ok,
            fail=len(fail),
            total_s=f"{total_elapsed:.2f}",
            avg_s=f"{avg:.2f}",
        )
        self.status.setText(msg)
        self.meta.setText(
            f"{_escape(msg)}<br>Log: <code>{_escape(str(glog.log_path()))}</code>"
        )
        self.meta.setTextFormat(Qt.TextFormat.RichText)

        if fail:
            sample = "\n".join(f"{r.note_id}: {r.error}" for r in fail[:6])
            hint = ""
            if any(
                "404" in (r.error or "") or "not found" in (r.error or "").lower()
                for r in fail
            ):
                hint = "\n\nModel may be unavailable — Refresh models and pick a listed one."
            QMessageBox.warning(
                self,
                "Gemini finished with errors",
                f"{msg}\n{len(fail)} failed:\n{sample}{hint}\n\nSee log:\n{glog.log_path()}",
            )
        elif not cancelled:
            tooltip(msg)

        if was_preview and results:
            self._show_test_result(results[0])
            self.btn_apply_preview.setEnabled(
                self._pending_preview is not None and self._pending_preview.ok
            )
            return

        keep_id = self._keep_note_id
        s = self._run_settings
        if not cancelled and not s.get("overwrite") and len(self._run_jobs) > 1:
            self._find(need_only=True)
        if keep_id is not None:
            self._select_note_in_list(keep_id)


def open_gemini_dialog(note_ids: list[int] | None = None) -> None:
    if mw.col is None:
        showInfo("Open a profile / collection first.")
        return
    dlg = GeminiDialog(mw, note_ids=note_ids)
    mw._gemini_sentence_dialog = dlg  # type: ignore[attr-defined]
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
