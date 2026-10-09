"""Derive Android and web branding from the desktop release icon and wordmark."""
from pathlib import Path
from shutil import copyfile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT.parent / 'adam-desktop' / 'resources'
WEB = ROOT / 'apps/web/public'
RES = ROOT / 'apps/mobile-shell/android/app/src/main/res'
icon = Image.open(DESKTOP / 'icons/adam.ico').convert('RGBA')
copyfile(DESKTOP / 'static/images/adam-wordmark.png', WEB / 'assets/adam-wordmark.png')
icon.save(WEB / 'assets/adam-logo.png')
for size in (192, 512):
    icon.resize((size, size), Image.Resampling.LANCZOS).save(WEB / f'icons/adam-{size}.png')
for density, scale in [('mdpi', 1), ('hdpi', 1.5), ('xhdpi', 2), ('xxhdpi', 3), ('xxxhdpi', 4)]:
    folder = RES / f'mipmap-{density}'
    size = int(48 * scale)
    regular = icon.resize((size, size), Image.Resampling.LANCZOS)
    regular.save(folder / 'ic_launcher.png')
    rounded = Image.new('RGBA', (size, size), '#101114')
    rounded.alpha_composite(regular)
    mask = Image.new('L', (size, size))
    ImageDraw.Draw(mask).ellipse((0, 0, size-1, size-1), fill=255)
    rounded.putalpha(mask)
    rounded.save(folder / 'ic_launcher_round.png')
    size, safe = int(108 * scale), int(66 * scale)
    foreground = Image.new('RGBA', (size, size))
    foreground.alpha_composite(icon.resize((safe, safe), Image.Resampling.LANCZOS), ((size-safe)//2, (size-safe)//2))
    foreground.save(folder / 'ic_launcher_foreground.png')
for target in RES.glob('drawable*/splash.png'):
    with Image.open(target) as previous:
        width, height = previous.size
    splash = Image.new('RGBA', (width, height), '#000000')
    size = round(min(width, height) * 0.3)
    splash.alpha_composite(icon.resize((size, size), Image.Resampling.LANCZOS), ((width-size)//2, (height-size)//2))
    splash.convert('RGB').save(target)
print('Desktop icon and wordmark applied to Android launcher, splash and web assets.')
