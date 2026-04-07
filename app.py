import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD


BG = "#0D0D0F"
SURFACE = "#141418"
SURFACE2 = "#1C1C22"
ACCENT = "#7C6AF7"
ACCENT2 = "#A78BFA"
SUCCESS = "#34D399"
ERROR = "#F87171"
TEXT = "#F0EFF8"
TEXT_DIM = "#6B6A7E"

VIDEO_FILE_TYPES = [
	("Video Files", "*.mp4 *.avi *.mov *.mkv"),
	("All Files", "*.*"),
]
MODEL_FILE_TYPES = [
	("Model Weights", "*.pth *.pt"),
	("All Files", "*.*"),
]
NO_RECENT = "No recent files"
RECENT_LIMIT = 5
RECENT_FILE = Path.home() / ".vfi_recent.json"


class DnDCTk(TkinterDnD.DnDWrapper, ctk.CTk):
	def __init__(self, *args, **kwargs):
		ctk.CTk.__init__(self, *args, **kwargs)
		TkinterDnD.DnDWrapper.__init__(self)


class VFIApp:
	def __init__(self) -> None:
		ctk.set_appearance_mode("dark")
		ctk.set_default_color_theme("dark-blue")

		self.root = DnDCTk()

		self.input_path_var = tk.StringVar(value="")
		self.output_path_var = tk.StringVar(value="")
		self.weights_path_var = tk.StringVar(value="models/best.pth")
		self.recent_var = tk.StringVar(value=NO_RECENT)
		self.recent_files = self._load_recent_files()
		self.root.title("VFI — Video Frame Interpolation")
		self.root.geometry("900x620")
		self.root.resizable(False, False)
		self.root.configure(fg_color=BG)

		self._center_window(900, 620)

		self.root.grid_rowconfigure(1, weight=1)
		self.root.grid_columnconfigure(0, weight=1)

		header = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=0, height=68)
		header.grid(row=0, column=0, sticky="nsew")
		header.grid_columnconfigure(0, weight=1)
		header.grid_columnconfigure(1, weight=0)

		title_label = ctk.CTkLabel(
			header,
			text="VFI",
			text_color=TEXT,
			font=ctk.CTkFont(size=32, weight="bold"),
		)
		title_label.grid(row=0, column=0, sticky="w", padx=20, pady=14)

		version_badge = ctk.CTkLabel(
			header,
			text="v3.0",
			text_color=TEXT,
			fg_color=ACCENT,
			corner_radius=12,
			padx=12,
			pady=5,
			font=ctk.CTkFont(size=13, weight="bold"),
		)
		version_badge.grid(row=0, column=1, sticky="e", padx=20, pady=18)

		content = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
		content.grid(row=1, column=0, sticky="nsew")
		content.grid_rowconfigure(0, weight=1)
		content.grid_columnconfigure(0, weight=0, minsize=340)
		content.grid_columnconfigure(1, weight=0, minsize=1)
		content.grid_columnconfigure(2, weight=1)

		left_panel = ctk.CTkFrame(content, fg_color=SURFACE, corner_radius=0, width=340)
		left_panel.grid(row=0, column=0, sticky="nsew")
		left_panel.grid_propagate(False)
		left_panel.grid_columnconfigure(0, weight=1)

		divider = ctk.CTkFrame(content, fg_color=SURFACE2, corner_radius=0, width=1)
		divider.grid(row=0, column=1, sticky="ns")

		right_panel = ctk.CTkFrame(content, fg_color=BG, corner_radius=0)
		right_panel.grid(row=0, column=2, sticky="nsew")

		settings_body = ctk.CTkFrame(left_panel, fg_color="transparent")
		settings_body.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 12))
		settings_body.grid_columnconfigure(0, weight=1)

		input_label = ctk.CTkLabel(
			settings_body,
			text="Input Video",
			text_color=TEXT,
			font=ctk.CTkFont(size=14, weight="bold"),
		)
		input_label.grid(row=0, column=0, sticky="w")

		self.input_entry = ctk.CTkEntry(
			settings_body,
			textvariable=self.input_path_var,
			height=34,
			fg_color=SURFACE2,
			text_color=TEXT,
			border_color=SURFACE2,
		)
		self.input_entry.grid(row=1, column=0, sticky="ew", pady=(8, 8))
		self.input_entry.bind("<Key>", lambda _event: "break")
		self.input_entry.bind("<<Paste>>", lambda _event: "break")

		input_browse_btn = ctk.CTkButton(
			settings_body,
			text="Browse",
			height=32,
			fg_color=ACCENT,
			hover_color=ACCENT2,
			text_color=TEXT,
			command=self._browse_input_video,
		)
		input_browse_btn.grid(row=2, column=0, sticky="ew", pady=(0, 18))

		output_label = ctk.CTkLabel(
			settings_body,
			text="Output File",
			text_color=TEXT,
			font=ctk.CTkFont(size=14, weight="bold"),
		)
		output_label.grid(row=3, column=0, sticky="w")

		self.output_entry = ctk.CTkEntry(
			settings_body,
			textvariable=self.output_path_var,
			height=34,
			fg_color=SURFACE2,
			text_color=TEXT,
			border_color=SURFACE2,
		)
		self.output_entry.grid(row=4, column=0, sticky="ew", pady=(8, 8))

		output_browse_btn = ctk.CTkButton(
			settings_body,
			text="Browse",
			height=32,
			fg_color=ACCENT,
			hover_color=ACCENT2,
			text_color=TEXT,
			command=self._browse_output_file,
		)
		output_browse_btn.grid(row=5, column=0, sticky="ew", pady=(0, 18))

		weights_label = ctk.CTkLabel(
			settings_body,
			text="Model Weights",
			text_color=TEXT,
			font=ctk.CTkFont(size=14, weight="bold"),
		)
		weights_label.grid(row=6, column=0, sticky="w")

		weights_row = ctk.CTkFrame(settings_body, fg_color="transparent")
		weights_row.grid(row=7, column=0, sticky="ew", pady=(8, 18))
		weights_row.grid_columnconfigure(0, weight=1)

		self.weights_entry = ctk.CTkEntry(
			weights_row,
			textvariable=self.weights_path_var,
			height=34,
			fg_color=SURFACE2,
			text_color=TEXT,
			border_color=SURFACE2,
		)
		self.weights_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

		weights_browse_btn = ctk.CTkButton(
			weights_row,
			text="...",
			width=44,
			height=34,
			fg_color=ACCENT,
			hover_color=ACCENT2,
			text_color=TEXT,
			command=self._browse_model_weights,
		)
		weights_browse_btn.grid(row=0, column=1, sticky="e")

		recent_label = ctk.CTkLabel(
			settings_body,
			text="Recent",
			text_color=TEXT,
			font=ctk.CTkFont(size=14, weight="bold"),
		)
		recent_label.grid(row=8, column=0, sticky="w")

		self.recent_menu = ctk.CTkOptionMenu(
			settings_body,
			variable=self.recent_var,
			values=self.recent_files or [NO_RECENT],
			fg_color=SURFACE2,
			button_color=ACCENT,
			button_hover_color=ACCENT2,
			text_color=TEXT,
			dropdown_fg_color=SURFACE2,
			dropdown_hover_color=SURFACE,
			dropdown_text_color=TEXT,
			command=self._on_recent_selected,
		)
		self.recent_menu.grid(row=9, column=0, sticky="ew", pady=(8, 0))
		self.recent_var.set(self.recent_files[0] if self.recent_files else NO_RECENT)

		self._setup_drag_and_drop()

		right_placeholder = ctk.CTkLabel(
			right_panel,
			text="Video Inspector coming soon",
			text_color=TEXT_DIM,
			font=ctk.CTkFont(size=18),
		)
		right_placeholder.place(relx=0.5, rely=0.5, anchor="center")

		bottom_bar = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=0, height=52)
		bottom_bar.grid(row=2, column=0, sticky="nsew")

		progress_placeholder = ctk.CTkLabel(
			bottom_bar,
			text="Progress bar placeholder",
			text_color=TEXT_DIM,
			font=ctk.CTkFont(size=14),
		)
		progress_placeholder.place(relx=0.5, rely=0.5, anchor="center")

	def _center_window(self, width: int, height: int) -> None:
		self.root.update_idletasks()
		screen_width = self.root.winfo_screenwidth()
		screen_height = self.root.winfo_screenheight()
		x = (screen_width - width) // 2
		y = (screen_height - height) // 2
		self.root.geometry(f"{width}x{height}+{x}+{y}")

	def _browse_input_video(self) -> None:
		selected = filedialog.askopenfilename(title="Select Input Video", filetypes=VIDEO_FILE_TYPES)
		if selected:
			self._set_input_path(selected)

	def _browse_output_file(self) -> None:
		selected = filedialog.asksaveasfilename(
			title="Select Output File",
			defaultextension=".mp4",
			filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
		)
		if selected:
			self.output_path_var.set(selected)

	def _browse_model_weights(self) -> None:
		selected = filedialog.askopenfilename(title="Select Model Weights", filetypes=MODEL_FILE_TYPES)
		if selected:
			self.weights_path_var.set(selected)

	def _set_input_path(self, input_path: str) -> None:
		clean_path = str(Path(input_path).expanduser())
		self.input_path_var.set(clean_path)
		self.output_path_var.set(self._default_output_from_input(clean_path))
		self._update_recent_files(clean_path)

	def _default_output_from_input(self, input_path: str) -> str:
		in_path = Path(input_path)
		return str(in_path.with_name(f"{in_path.stem}_interpolated.mp4"))

	def _setup_drag_and_drop(self) -> None:
		self.input_entry.drop_target_register(DND_FILES)
		self.input_entry.dnd_bind("<<Drop>>", self._on_input_drop)

	def _on_input_drop(self, event) -> None:
		files = self.root.tk.splitlist(event.data)
		if not files:
			return
		first_file = str(files[0]).strip("{}")
		if first_file:
			self._set_input_path(first_file)

	def _load_recent_files(self) -> list[str]:
		if not RECENT_FILE.exists():
			return []
		try:
			data = json.loads(RECENT_FILE.read_text(encoding="utf-8"))
		except (json.JSONDecodeError, OSError):
			return []
		if not isinstance(data, list):
			return []
		result = [str(item) for item in data if isinstance(item, str) and item.strip()]
		return result[:RECENT_LIMIT]

	def _save_recent_files(self) -> None:
		try:
			RECENT_FILE.write_text(json.dumps(self.recent_files, indent=2), encoding="utf-8")
		except OSError:
			return

	def _update_recent_files(self, input_path: str) -> None:
		self.recent_files = [path for path in self.recent_files if path != input_path]
		self.recent_files.insert(0, input_path)
		self.recent_files = self.recent_files[:RECENT_LIMIT]
		self._save_recent_files()
		self._refresh_recent_menu()

	def _refresh_recent_menu(self) -> None:
		values = self.recent_files if self.recent_files else [NO_RECENT]
		self.recent_menu.configure(values=values)
		self.recent_var.set(values[0])

	def _on_recent_selected(self, selected: str) -> None:
		if selected == NO_RECENT:
			return
		self._set_input_path(selected)

	def run(self) -> None:
		self.root.mainloop()


if __name__ == "__main__":
	VFIApp().run()
