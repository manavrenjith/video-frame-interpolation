import customtkinter as ctk


BG = "#0D0D0F"
SURFACE = "#141418"
SURFACE2 = "#1C1C22"
ACCENT = "#7C6AF7"
ACCENT2 = "#A78BFA"
SUCCESS = "#34D399"
ERROR = "#F87171"
TEXT = "#F0EFF8"
TEXT_DIM = "#6B6A7E"


class VFIApp:
	def __init__(self) -> None:
		ctk.set_appearance_mode("dark")
		ctk.set_default_color_theme("dark-blue")

		self.root = ctk.CTk()
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

		divider = ctk.CTkFrame(content, fg_color=SURFACE2, corner_radius=0, width=1)
		divider.grid(row=0, column=1, sticky="ns")

		right_panel = ctk.CTkFrame(content, fg_color=BG, corner_radius=0)
		right_panel.grid(row=0, column=2, sticky="nsew")

		left_placeholder = ctk.CTkLabel(
			left_panel,
			text="Settings coming in next increment",
			text_color=TEXT_DIM,
			font=ctk.CTkFont(size=16),
		)
		left_placeholder.place(relx=0.5, rely=0.5, anchor="center")

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

	def run(self) -> None:
		self.root.mainloop()


if __name__ == "__main__":
	VFIApp().run()
