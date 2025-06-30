import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

def cartoonify(img):
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7
    )
   color = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
   color = cv2.convertScaleAbs(color, alpha=1.3, beta=10)
   return cv2.bitwise_and(color, color, mask=edges)

def black_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def comic_style(img):
    img_up = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 2)
    color = cv2.bilateralFilter(img_up, d=9, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cv2.resize(cartoon, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def crayon_sketch(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    def posterize(img, levels=6):
        div = 256 // levels
        return (img // div) * div

    poster = posterize(color)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    sketch = cv2.bitwise_and(poster, edges_colored)
    bright = cv2.convertScaleAbs(sketch, alpha=1.5, beta=50)
    return bright

def laplacian_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
    inv_lap = 255 - laplacian
    norm_lap = cv2.normalize(inv_lap, None, 0, 255, cv2.NORM_MINMAX)
    lap_sketch = cv2.cvtColor(norm_lap, cv2.COLOR_GRAY2BGR)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    pencil = cv2.divide(gray, inverted_blurred, scale=256.0)
    pencil_bgr = cv2.cvtColor(pencil, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(lap_sketch,1.0, pencil_bgr, 0.0, 0)
    return blended

def instant_collage(image_list, effect_func):
    processed = []
    size = (200, 200)
    for img in image_list:
        img_proc = effect_func(img)
        img_proc = cv2.resize(img_proc, size)
        processed.append(img_proc)
    n = len(processed)
    if n == 0:
        return None
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    blank = np.ones_like(processed[0]) * 255
    while len(processed) < rows * cols:
        processed.append(blank)
    grid = []
    for r in range(rows):
        row_imgs = processed[r*cols:(r+1)*cols]
        grid.append(np.hstack(row_imgs))
    collage = np.vstack(grid)
    return collage

class CartoonifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ARTIFY")
        self.image = None
        self.orig_image = None
        self.processed = None

        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10)

        self.img_panel = tk.Label(frame)
        self.img_panel.grid(row=0, column=0, columnspan=4)

        tk.Label(frame, text="Effect:").grid(row=1, column=0, sticky="e")
        self.effect_var = tk.StringVar(value="Cartoonify")
        effects = [
            "Cartoonify",
            "Binary Ink",
            "Realistic InkToon",
            "Poster Toon",
            "Pencil Line Mono"
        ]
        self.effect_menu = ttk.Combobox(frame, textvariable=self.effect_var, values=effects, state="readonly", width=20)
        self.effect_menu.grid(row=1, column=1, sticky="w")

        tk.Label(frame, text="Edge Threshold:").grid(row=2, column=0, sticky="e")
        self.edge_slider = tk.Scale(frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.edge_slider.set(7)
        self.edge_slider.grid(row=2, column=1, sticky="w")

        tk.Label(frame, text="Color Smoothing:").grid(row=2, column=2, sticky="e")
        self.smooth_slider = tk.Scale(frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.smooth_slider.set(7)
        self.smooth_slider.grid(row=2, column=3, sticky="w")

        tk.Button(frame, text="Open Image", command=self.open_image).grid(row=3, column=0, pady=10)
        tk.Button(frame, text="Apply Effect", command=self.apply_effect).grid(row=3, column=1)
        tk.Button(frame, text="Save Result", command=self.save_image).grid(row=3, column=2)
        tk.Button(frame, text="Reset", command=self.reset_image).grid(row=3, column=3)
        tk.Button(frame, text="Capture from Camera", command=self.capture_from_camera).grid(row=4, column=0, columnspan=4, pady=5)
        tk.Button(frame, text="Instant Collage", command=self.instant_collage_ui).grid(row=5, column=0, columnspan=4, pady=5)
   
    def capture_from_camera(self):
     cap = cv2.VideoCapture(0)
     if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera.")
        return
     messagebox.showinfo("Camera", "Press SPACE to capture, ESC to cancel.")
     while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32: 
            self.image = frame.copy()
            self.orig_image = frame.copy()
            self.processed = None
            self.show_image(self.image, self.image)
            break
     cap.release()
     cv2.destroyAllWindows()
     
    def instant_collage_ui(self):
      paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
      if not paths:
        return
      images = []
      for path in paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
      if not images:
        messagebox.showerror("Error", "No valid images selected.")
        return
      effect = self.effect_var.get()
      if effect == "Cartoonify":
        func = cartoonify
      elif effect == "Binary Ink":
        func = black_white
      elif effect == "Realistic InkToon":
        func = comic_style
      elif effect == "Poster Toon":
        func = crayon_sketch
      elif effect == "Pencil Line Mono":
         func = laplacian_sketch
      else:
        func = cartoonify
      collage = instant_collage(images, func)
      if collage is not None:
        self.processed = collage
        self.show_image(collage, collage)
   
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.image = img.copy()
                self.orig_image = img.copy()
                self.processed = None
                self.show_image(self.image, self.image)
            else:
                messagebox.showerror("Error", "Could not read the image file.")

    def apply_effect(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        effect = self.effect_var.get()
        edge_val = self.edge_slider.get()
        smooth_val = self.smooth_slider.get()
        if effect == "Cartoonify":
            self.processed = cartoonify(self.image)
        elif effect == "Binary Ink":
            self.processed = black_white(self.image)
        elif effect == "Realistic InkToon":
            self.processed = comic_style(self.image)
        elif effect == "Poster Toon":
            self.processed = crayon_sketch(self.image)
        elif effect == "Pencil Line Mono":
            self.processed = laplacian_sketch(self.image)
        self.show_image(self.image, self.processed)

    def save_image(self):
        if self.processed is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.processed)
                messagebox.showinfo("Saved", f"Image saved as {path}")

    def reset_image(self):
        if self.orig_image is not None:
            self.image = self.orig_image.copy()
            self.processed = None
            self.show_image(self.image, self.image)

    def show_image(self, img1, img2):
        img1_disp = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_disp = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2 is not None else img1_disp
        h = max(img1_disp.shape[0], img2_disp.shape[0])
        w = img1_disp.shape[1] + img2_disp.shape[1]
        combined = np.ones((h, w, 3), dtype=np.uint8) * 255
        combined[:img1_disp.shape[0], :img1_disp.shape[1]] = img1_disp
        combined[:img2_disp.shape[0], img1_disp.shape[1]:] = img2_disp
        img_pil = Image.fromarray(combined)
        img_pil = img_pil.resize((900,450))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_panel.config(image=img_tk)
        self.img_panel.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = CartoonifyApp(root)
    root.mainloop()
