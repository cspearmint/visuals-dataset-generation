"""
Lightweight image degradations for synthetic weather/lighting effects.

Effects provided:
- rain
- fog
- snow
- motion blur
- low light (nighttime)

Designed to be dependency-light (Pillow + stdlib).
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional
import random

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


def decode_image(data: bytes) -> Image.Image:
    """Decode raw image bytes into a PIL Image (RGB)."""
    return Image.open(BytesIO(data)).convert("RGB")


def encode_image(image: Image.Image, format: str = "JPEG", quality: int = 90) -> bytes:
    """Encode a PIL Image back to bytes."""
    buf = BytesIO()
    image.save(buf, format=format, quality=quality)
    return buf.getvalue()


def _gamma_lut(gamma: float):
    """Build 256-entry gamma LUT for Image.point()."""
    inv = 1.0 / max(gamma, 1e-6)
    return [int(((i / 255.0) ** inv) * 255.0 + 0.5) for i in range(256)]


def _scale_lut(scale: float):
    """Scale LUT (clamped) for Image.point()."""
    s = max(0.0, scale)
    return [max(0, min(255, int(i * s + 0.5))) for i in range(256)]


def _contrast_compress(img: Image.Image, factor: float) -> Image.Image:
    """
    Reduce contrast by pulling values toward the global mean.
    factor < 1 compresses contrast.
    """
    return ImageEnhance.Contrast(img).enhance(factor)


def _desaturate(img: Image.Image, factor: float) -> Image.Image:
    """Reduce saturation (factor < 1)."""
    return ImageEnhance.Color(img).enhance(factor)


@dataclass
class ImageAugmenter:
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -------- Public API --------
    def apply_all(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Apply all supported degradations."""
        rng = self._rng
        return {
            "rain": self.rain(image, rng),
            "fog": self.fog(image, rng),
            "snow": self.snow(image, rng),
            "motion_blur": self.motion_blur(image),
            "low_light": self.low_light(image),
        }

    # -------- Effects --------
    def rain(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """
        More realistic rain (still Pillow-only):
        - thin motion-blurred streaks that DARKEN (not brighten) the scene
        - global darkening + slight contrast loss + slight desaturation
        - subtle atmospheric haze
        """
        img = image.convert("RGB")
        w, h = img.size

        # --- 1) Streak mask (L) ---
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Density proportional to area, with bounds.
        drops = max(350, (w * h) // 6000)
        slant = rng.randint(-14, 14)
        min_len = max(14, min(26, h // 60))
        max_len = max(min_len + 6, min(48, h // 18))

        for _ in range(drops):
            x = rng.randint(-w // 10, w + w // 10)
            y = rng.randint(-h // 10, h + h // 10)
            length = rng.randint(min_len, max_len)
            # mostly 1px; sometimes 2px
            thickness = 1 if rng.random() < 0.85 else 2
            # intensity controls how much darkening shows up
            intensity = rng.randint(140, 210)
            draw.line((x, y, x + slant, y + length), fill=intensity, width=thickness)

        # Blur to simulate motion/defocus
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.1))

        # Thin streaks (raise contrast / gamma-like curve)
        # Use a LUT to avoid per-pixel Python.
        mask = mask.point(_gamma_lut(gamma=1.7))

        # Increased rain intensity
        mask = mask.point(_scale_lut(0.85))  # Was 0.60, now more intense

        # --- 2) Darken streak regions more ---
        dark = ImageEnhance.Brightness(img).enhance(0.60)  # Was 0.75, now darker
        streaked = Image.composite(dark, img, mask)

        # --- 3) Global rainy atmosphere (more pronounced) ---
        # Stronger global darkening
        streaked = ImageEnhance.Brightness(streaked).enhance(0.85)  # Was 0.93
        # Stronger contrast reduction
        streaked = _contrast_compress(streaked, 0.75)  # Was 0.90
        # Stronger desaturation
        streaked = _desaturate(streaked, 0.70)  # Was 0.85

        # --- 4) More visible haze ---
        # Use Pillow's built-in noise generator (fast, no numpy).
        haze = Image.effect_noise((w, h), sigma=18).convert("L")
        haze = haze.filter(ImageFilter.GaussianBlur(radius=14))
        haze = haze.point(_scale_lut(0.25))  # Was 0.10, now 2.5x more visible

        haze_color = Image.new("RGB", (w, h), (230, 230, 230))
        out = Image.composite(haze_color, streaked, haze)

        return out

    def fog(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """
        More realistic dense atmospheric fog (Pillow-only):
        - low-frequency fog density field (no white blob ellipses)
        - blend toward light gray (scattering)
        - contrast compression + slight desaturation
        - gentle detail preservation (very mild unsharp mask)
        """
        img = image.convert("RGB")
        w, h = img.size

        # --- 1) Low-frequency fog density field ---
        # Start with noise in [0,255]
        noise = Image.effect_noise((w, h), sigma=32).convert("L")

        # Large blur kernel to make it low-frequency
        # Radius scales with image size but bounded.
        radius = max(18, min(42, min(w, h) // 28))
        density = noise.filter(ImageFilter.GaussianBlur(radius=radius))

        # Bias toward heavier fog while remaining plausible:
        # - increase mid/high densities via gamma
        density = density.point(_gamma_lut(gamma=0.65))  # gamma < 1 => heavier

        # Increased fog strength (much denser)
        strength = rng.uniform(0.65, 0.80)  # Was 0.45-0.60, now much stronger
        density = density.point(_scale_lut(strength))

        # --- 2) Atmospheric scattering blend toward light gray ---
        fog_color = Image.new("RGB", (w, h), (235, 235, 235))
        fogged = Image.composite(fog_color, img, density)

        # --- 3) Fog kills contrast + saturation (more aggressive) ---
        fogged = _contrast_compress(fogged, 0.65)  # Was 0.82
        fogged = _desaturate(fogged, 0.60)  # Was 0.78

        # --- 4) Gentle detail preservation (mild unsharp) ---
        blur = fogged.filter(ImageFilter.GaussianBlur(radius=1.1))
        # Blend back a small amount of high-frequency detail
        out = Image.blend(fogged, ImageChops.subtract(fogged, blur).filter(ImageFilter.GaussianBlur(0.3)), 0.15)
        # The subtraction image is centered near black; blending can darken slightly.
        # Counter with a tiny brightness lift (very mild).
        out = ImageEnhance.Brightness(out).enhance(1.02)

        return out

    def snow(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Overlay snowflakes as small white dots."""
        img = image.convert("RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        flakes = max(300, (w * h) // 6000)
        for _ in range(flakes):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            r = rng.randint(1, 3)
            alpha = rng.randint(150, 255)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, alpha))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
        combined = Image.alpha_composite(img, overlay)
        return combined.convert("RGB")

    def motion_blur(self, image: Image.Image) -> Image.Image:
        """Approximate horizontal motion blur without custom kernels."""
        blur = image.filter(ImageFilter.BoxBlur(3))
        shifted = ImageChops.offset(image, 5, 0)
        return Image.blend(blur, shifted, alpha=0.5)

    def low_light(self, image: Image.Image) -> Image.Image:
        """Dim and desaturate to simulate nighttime/low light."""
        dark = ImageEnhance.Brightness(image).enhance(0.3)
        color = ImageEnhance.Color(dark).enhance(0.7)
        return color
