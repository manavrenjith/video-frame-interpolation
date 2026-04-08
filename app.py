import json
from pathlib import Path
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image
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
THUMBNAIL_MAX_W = 400
THUMBNAIL_MAX_H = 250


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
		self.thumbnail_image = None
		self._inspector_request_id = 0
		self._inspector_loading = False
		self._processing_active = False
		self._cancel_event = threading.Event()
		self._worker_thread = None
		self._process_start_ts = 0.0
		self._run_button_reset_id = None

		self.filename_var = tk.StringVar(value="-")
		self.duration_var = tk.StringVar(value="-")
		self.resolution_var = tk.StringVar(value="-")
		self.current_fps_var = tk.StringVar(value="-")
		self.frame_count_var = tk.StringVar(value="-")
		self.output_fps_var = tk.StringVar(value="-")
		self.output_frames_var = tk.StringVar(value="-")
		self.inspector_status_var = tk.StringVar(value="")
		self.progress_status_var = tk.StringVar(value="Idle")
		self.percent_var = tk.StringVar(value="0%")
		self.eta_var = tk.StringVar(value="ETA: --")
		self.root.title("VFI — Video Frame Interpolation")
		self.root.geometry("900x620")
		self.root.resizable(False, False)
		self.root.configure(fg_color=BG)

		self._center_window(900, 620)

		self.root.grid_rowconfigure(1, weight=1)
		self.root.grid_rowconfigure(2, weight=0)
		self.root.grid_rowconfigure(3, weight=0)
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
		left_panel.grid_rowconfigure(0, weight=1)

		divider = ctk.CTkFrame(content, fg_color=SURFACE2, corner_radius=0, width=1)
		divider.grid(row=0, column=1, sticky="ns")

		right_panel = ctk.CTkFrame(content, fg_color=BG, corner_radius=0)
		right_panel.grid(row=0, column=2, sticky="nsew")
		right_panel.grid_columnconfigure(0, weight=1)
		right_panel.grid_rowconfigure(0, weight=1)

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

		actions_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
		actions_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 16))
		actions_frame.grid_columnconfigure(0, weight=1)

		self.run_button = ctk.CTkButton(
			actions_frame,
			text="▶  Run Interpolation",
			height=38,
			fg_color=ACCENT,
			hover_color=ACCENT2,
			text_color=TEXT,
			command=self._on_run_clicked,
		)
		self.run_button.grid(row=0, column=0, sticky="ew")

		self.cancel_button = ctk.CTkButton(
			actions_frame,
			text="Cancel",
			height=34,
			fg_color=SURFACE2,
			hover_color="#2A2A32",
			text_color=TEXT,
			state="disabled",
			command=self._on_cancel_clicked,
		)
		self.cancel_button.grid(row=1, column=0, sticky="ew", pady=(8, 0))

		self._setup_drag_and_drop()
		self._build_video_inspector(right_panel)

		log_pane = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=0, height=100)
		log_pane.grid(row=2, column=0, sticky="nsew")
		log_pane.grid_propagate(False)
		log_pane.grid_columnconfigure(0, weight=1)

		self.log_box = ctk.CTkTextbox(
			log_pane,
			height=100,
			fg_color="#0E0E12",
			text_color=TEXT,
			font=ctk.CTkFont(family="Consolas", size=12),
			border_width=1,
			border_color=SURFACE2,
			wrap="word",
		)
		self.log_box.grid(row=0, column=0, sticky="nsew", padx=12, pady=8)
		self.log_box.configure(state="disabled")

		bottom_bar = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=0, height=56)
		bottom_bar.grid(row=3, column=0, sticky="nsew")
		bottom_bar.grid_columnconfigure(0, weight=1)
		bottom_bar.grid_columnconfigure(1, weight=0)
		bottom_bar.grid_columnconfigure(2, weight=0)

		self.progress_bar = ctk.CTkProgressBar(bottom_bar, mode="determinate", height=14)
		self.progress_bar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 4))
		self.progress_bar.set(0)

		self.progress_status_label = ctk.CTkLabel(
			bottom_bar,
			textvariable=self.progress_status_var,
			text_color=TEXT_DIM,
			font=ctk.CTkFont(size=13, weight="bold"),
		)
		self.progress_status_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

		self.percent_label = ctk.CTkLabel(
			bottom_bar,
			textvariable=self.percent_var,
			text_color=TEXT,
			font=ctk.CTkFont(size=13, weight="bold"),
		)
		self.percent_label.grid(row=1, column=1, sticky="e", padx=(0, 14), pady=(0, 8))

		self.eta_label = ctk.CTkLabel(
			bottom_bar,
			textvariable=self.eta_var,
			text_color=TEXT_DIM,
			font=ctk.CTkFont(size=13),
		)
		self.eta_label.grid(row=1, column=2, sticky="e", padx=(0, 12), pady=(0, 8))

		self._set_processing_active(False)

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
		self._start_video_inspector_load(clean_path)

	def _default_output_from_input(self, input_path: str) -> str:
		in_path = Path(input_path)
		return str(in_path.with_name(f"{in_path.stem}_interpolated.mp4"))

	def _setup_drag_and_drop(self) -> None:
		try:
			self.input_entry.drop_target_register(DND_FILES)
			self.input_entry.dnd_bind("<<Drop>>", self._on_input_drop)
		except tk.TclError:
			# Some Python/Tk builds do not include tkdnd binaries.
			return

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

	def _build_video_inspector(self, parent: ctk.CTkFrame) -> None:
		inspector_card = ctk.CTkFrame(
			parent,
			fg_color=SURFACE,
			border_width=1,
			border_color=SURFACE2,
			corner_radius=10,
		)
		inspector_card.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
		inspector_card.grid_columnconfigure(0, weight=1)

		card_title = ctk.CTkLabel(
			inspector_card,
			text="Video Inspector",
			text_color=TEXT,
			font=ctk.CTkFont(size=18, weight="bold"),
		)
		card_title.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 10))

		self.thumbnail_frame = ctk.CTkFrame(
			inspector_card,
			width=THUMBNAIL_MAX_W,
			height=THUMBNAIL_MAX_H,
			fg_color="#2B2B31",
			corner_radius=8,
		)
		self.thumbnail_frame.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="n")
		self.thumbnail_frame.grid_propagate(False)

		self.thumbnail_label = ctk.CTkLabel(
			self.thumbnail_frame,
			text="Loading...",
			text_color=TEXT_DIM,
			image=None,
		)
		self.thumbnail_label.place(relx=0.5, rely=0.5, anchor="center")

		self.inspector_progress = ctk.CTkProgressBar(inspector_card, mode="indeterminate")
		self.inspector_progress.grid(row=2, column=0, sticky="ew", padx=16)
		self.inspector_progress.grid_remove()

		self.inspector_status_label = ctk.CTkLabel(
			inspector_card,
			textvariable=self.inspector_status_var,
			text_color=ERROR,
			font=ctk.CTkFont(size=13, weight="bold"),
		)
		self.inspector_status_label.grid(row=3, column=0, sticky="w", padx=16, pady=(8, 2))

		meta_grid = ctk.CTkFrame(inspector_card, fg_color="transparent")
		meta_grid.grid(row=4, column=0, sticky="nsew", padx=16, pady=(8, 14))
		meta_grid.grid_columnconfigure(0, weight=0)
		meta_grid.grid_columnconfigure(1, weight=1)

		rows = [
			("Filename", self.filename_var),
			("Duration", self.duration_var),
			("Resolution", self.resolution_var),
			("Current FPS", self.current_fps_var),
			("Frame Count", self.frame_count_var),
			("Output FPS", self.output_fps_var),
			("Output Frames", self.output_frames_var),
		]
		for row_index, (label_text, var) in enumerate(rows):
			label = ctk.CTkLabel(
				meta_grid,
				text=f"{label_text}:",
				text_color=TEXT_DIM,
				font=ctk.CTkFont(size=13),
			)
			label.grid(row=row_index, column=0, sticky="w", pady=2)

			value = ctk.CTkLabel(
				meta_grid,
				textvariable=var,
				text_color=TEXT,
				font=ctk.CTkFont(size=13, weight="bold"),
			)
			value.grid(row=row_index, column=1, sticky="w", padx=(10, 0), pady=2)

	def _start_video_inspector_load(self, input_path: str) -> None:
		self._inspector_request_id += 1
		request_id = self._inspector_request_id
		self._set_inspector_loading(True)
		self.inspector_status_var.set("")
		self._set_thumbnail_loading()

		worker = threading.Thread(
			target=self._load_video_inspector_worker,
			args=(request_id, input_path),
			daemon=True,
		)
		worker.start()

	def _set_inspector_loading(self, loading: bool) -> None:
		self._inspector_loading = loading
		if loading:
			self.inspector_progress.grid()
			self.inspector_progress.start()
		else:
			self.inspector_progress.stop()
			self.inspector_progress.grid_remove()

	def _set_thumbnail_loading(self) -> None:
		self.thumbnail_image = None
		self.thumbnail_label.configure(image=None, text="Loading...", text_color=TEXT_DIM)

	def _load_video_inspector_worker(self, request_id: int, input_path: str) -> None:
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			cap.release()
			self.root.after(0, lambda: self._on_video_load_failed(request_id))
			return

		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

		ok, frame = cap.read()
		cap.release()
		if not ok or frame is None:
			self.root.after(0, lambda: self._on_video_load_failed(request_id))
			return

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)
		thumb_size = self._fit_inside(image.width, image.height, THUMBNAIL_MAX_W, THUMBNAIL_MAX_H)
		thumb = image.resize(thumb_size, Image.Resampling.LANCZOS)

		duration_seconds = (frame_count / fps) if fps > 0 else 0.0
		minutes = int(duration_seconds // 60)
		seconds = int(duration_seconds % 60)

		metadata = {
			"filename": Path(input_path).name,
			"duration": f"{minutes:02d}:{seconds:02d}",
			"resolution": f"{width} × {height}",
			"current_fps": f"{fps:.1f}",
			"frame_count": str(frame_count),
			"output_fps": f"{fps * 2:.1f}",
			"output_frames": str(frame_count * 2),
		}

		self.root.after(0, lambda: self._on_video_load_success(request_id, thumb, metadata))

	def _fit_inside(self, src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
		if src_w <= 0 or src_h <= 0:
			return max_w, max_h
		scale = min(max_w / src_w, max_h / src_h)
		return max(1, int(src_w * scale)), max(1, int(src_h * scale))

	def _on_video_load_success(self, request_id: int, thumb: Image.Image, metadata: dict[str, str]) -> None:
		if request_id != self._inspector_request_id:
			return
		self._set_inspector_loading(False)
		self.inspector_status_var.set("")

		self.thumbnail_image = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
		self.thumbnail_label.configure(image=self.thumbnail_image, text="")

		self.filename_var.set(metadata["filename"])
		self.duration_var.set(metadata["duration"])
		self.resolution_var.set(metadata["resolution"])
		self.current_fps_var.set(metadata["current_fps"])
		self.frame_count_var.set(metadata["frame_count"])
		self.output_fps_var.set(metadata["output_fps"])
		self.output_frames_var.set(metadata["output_frames"])

	def _on_video_load_failed(self, request_id: int) -> None:
		if request_id != self._inspector_request_id:
			return
		self._set_inspector_loading(False)
		self.inspector_status_var.set("Could not read video file")
		self._set_thumbnail_loading()
		self.filename_var.set("-")
		self.duration_var.set("-")
		self.resolution_var.set("-")
		self.current_fps_var.set("-")
		self.frame_count_var.set("-")
		self.output_fps_var.set("-")
		self.output_frames_var.set("-")

	def _on_run_clicked(self) -> None:
		if self._processing_active:
			return

		input_path = Path(self.input_path_var.get().strip()).expanduser()
		output_path = Path(self.output_path_var.get().strip()).expanduser()
		model_path = Path(self.weights_path_var.get().strip()).expanduser()

		errors: list[str] = []
		if not input_path.is_file():
			errors.append(f"Invalid input path: {input_path}")
		if not output_path.name:
			errors.append("Output path is empty")
		elif not output_path.parent.exists():
			errors.append(f"Output folder does not exist: {output_path.parent}")
		if not model_path.is_file():
			errors.append(f"Invalid model path: {model_path}")

		if errors:
			for message in errors:
				self._append_log(message)
			self._flash_run_button_red()
			return

		self.input_path_var.set(str(input_path))
		self.output_path_var.set(str(output_path))
		self.weights_path_var.set(str(model_path))
		self._cancel_event = threading.Event()
		self._process_start_ts = time.perf_counter()
		self.progress_bar.start()
		self.progress_bar.set(0)
		self.progress_bar.stop()
		self.progress_status_var.set("Starting...")
		self.progress_status_label.configure(text_color=TEXT)
		self.percent_var.set("0%")
		self.eta_var.set("ETA: --")
		self._append_log("Starting interpolation")
		self._set_processing_active(True)

		self._worker_thread = threading.Thread(
			target=self._worker,
			args=(str(input_path), str(output_path), str(model_path)),
			daemon=True,
		)
		self._worker_thread.start()

	def _on_cancel_clicked(self) -> None:
		if not self._processing_active:
			return
		self._cancel_event.set()
		self.progress_status_var.set("Cancelling...")
		self.progress_status_label.configure(text_color=TEXT_DIM)
		self._append_log("Cancellation requested")

	def _worker(self, input_video: str, output_video: str, model_path: str) -> None:
		from src.predict import interpolate_video

		def progress_cb(done: int, total: int) -> None:
			if self._cancel_event.is_set():
				raise InterruptedError("Cancelled")
			self.root.after(0, lambda d=done, t=total: self._on_progress(d, t))

		try:
			if self._cancel_event.is_set():
				raise InterruptedError("Cancelled")

			try:
				interpolate_video(
					input_video,
					output_video,
					model_path,
					factor=2,
					progress_cb=progress_cb,
					cancel_flag=self._cancel_event,
				)
			except TypeError:
				interpolate_video(input_video, output_video, model_path, 2)

			if self._cancel_event.is_set():
				raise InterruptedError("Cancelled")
		except InterruptedError:
			self.root.after(0, self._on_worker_cancelled)
			return
		except Exception:
			message = traceback.format_exc().rstrip()
			self.root.after(0, lambda msg=message: self._on_worker_error(msg))
			return

		self.root.after(0, lambda: self._on_worker_done(output_video))

	def _on_progress(self, done: int, total: int) -> None:
		if total <= 0:
			return
		done = max(0, min(done, total))
		progress = done / total
		self.progress_bar.set(progress)
		self.progress_status_var.set(f"Frame {done}/{total}")
		self.progress_status_label.configure(text_color=TEXT)
		self.percent_var.set(f"{int(progress * 100)}%")

		if done > (total * 0.05):
			elapsed = max(0.001, time.perf_counter() - self._process_start_ts)
			remaining = (elapsed / done) * (total - done)
			self.eta_var.set(f"ETA: {self._format_eta(remaining)}")
		else:
			self.eta_var.set("ETA: --")

	def _on_worker_done(self, output_video: str) -> None:
		self._set_processing_active(False)
		self.progress_bar.set(1.0)
		self.percent_var.set("100%")
		self.eta_var.set("ETA: 0s")
		self.progress_status_var.set("Done ✓")
		self.progress_status_label.configure(text_color=SUCCESS)
		self._append_log(f"Done. Output written to: {output_video}")

	def _on_worker_cancelled(self) -> None:
		self._set_processing_active(False)
		self.progress_status_var.set("Cancelled")
		self.progress_status_label.configure(text_color=TEXT_DIM)
		self._append_log("Cancelled")

	def _on_worker_error(self, message: str) -> None:
		self._set_processing_active(False)
		self.progress_status_var.set("Error")
		self.progress_status_label.configure(text_color=ERROR)
		self._append_log(message)

	def _set_processing_active(self, active: bool) -> None:
		self._processing_active = active
		self.run_button.configure(state="disabled" if active else "normal")
		self.cancel_button.configure(state="normal" if active else "disabled")

	def _flash_run_button_red(self) -> None:
		if self._run_button_reset_id is not None:
			self.root.after_cancel(self._run_button_reset_id)
		self.run_button.configure(fg_color=ERROR, hover_color=ERROR)

		def _reset() -> None:
			self._run_button_reset_id = None
			self.run_button.configure(fg_color=ACCENT, hover_color=ACCENT2)

		self._run_button_reset_id = self.root.after(700, _reset)

	def _append_log(self, message: str) -> None:
		elapsed = 0.0
		if self._process_start_ts > 0:
			elapsed = max(0.0, time.perf_counter() - self._process_start_ts)
		minutes = int(elapsed // 60)
		seconds = elapsed - (minutes * 60)
		prefix = f"[{minutes:02d}:{seconds:04.1f}]"

		lines = message.splitlines() or [message]
		self.log_box.configure(state="normal")
		for line in lines:
			self.log_box.insert("end", f"{prefix} {line}\n")
		self.log_box.see("end")
		self.log_box.configure(state="disabled")

	def _format_eta(self, seconds: float) -> str:
		seconds_int = max(0, int(round(seconds)))
		minutes, rem = divmod(seconds_int, 60)
		if minutes > 0:
			return f"{minutes}m {rem}s"
		return f"{rem}s"

	def run(self) -> None:
		self.root.mainloop()


if __name__ == "__main__":
	VFIApp().run()
