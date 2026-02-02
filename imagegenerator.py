import tkinter
import customtkinter as ctk
from PIL import ImageTk
import torch
from diffusers import StableDiffusionPipeline
import threading

# App theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.default_window_width = 1200
        self.default_window_height = 800
        

        self.title("Image Generator")
        self.geometry(f"{self.default_window_width}x{self.default_window_height}")

        # Window Title
        self.windowlabel = ctk.CTkLabel(
            self,
            text="Text to Image Generator",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="white"
        )
        self.windowlabel.pack(pady=20)

        # Prompt Entry
        self.promptentry = ctk.CTkEntry(
            self,
            placeholder_text="Enter your prompt here",
            width=self.default_window_width - 50,
            height=40
        )
        self.promptentry.pack(pady=20)

        # Generate Button
        self.generatebutton = ctk.CTkButton(
            master=self,
            text="Generate Image",
            command=self.start_generation
        )
        self.generatebutton.pack(pady=10)

    def start_generation(self):
        self.textprompt = self.promptentry.get()
        self.generatebutton.configure(state="disabled")

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.pack(pady=20)
        self.progress.start()

        threading.Thread(target=self.generate, daemon=True).start()

    def generate(self):
        modelid = "runwayml/stable-diffusion-v1-5"   # ✅ lighter + better quality
        device = torch.device("cpu")  # CPU only

        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            modelid,
            torch_dtype=torch.float32,
            token=self.authorization_token or None
        ).to(device)

        # Generate image
        image = pipe(self.textprompt, guidance_scale=7.5).images[0]
        image.save("generated.png")

        # Show image in UI
        self.img = ImageTk.PhotoImage(image)
        self.imageview = ctk.CTkLabel(self, image=self.img, width=600, height=400)
        self.imageview.image = self.img  # prevent garbage collection
        self.imageview.pack(pady=20)

        # Reset UI
        self.progress.stop()
        self.progress.pack_forget()
        self.generatebutton.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
