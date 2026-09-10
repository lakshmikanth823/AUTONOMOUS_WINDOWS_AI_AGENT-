"""Windows Native OCR tool using Windows.Media.Ocr WinRT engine without external dependencies."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

# OCR Status Codes
OCR_SUCCESS_TEXT_FOUND = "OCR_SUCCESS_TEXT_FOUND"
OCR_SUCCESS_NO_TEXT = "OCR_SUCCESS_NO_TEXT"
OCR_ENGINE_UNAVAILABLE = "OCR_ENGINE_UNAVAILABLE"
OCR_FAILED = "OCR_FAILED"


class OCRWord(BaseModel):
    """Single recognized word with bounding box and center coordinates."""

    text: str
    rect: Dict[str, int] = Field(description="Bounding box: left, top, right, bottom, width, height")
    center: Tuple[int, int] = Field(description="Center point (x, y)")
    confidence: Optional[float] = Field(default=None, description="Recognition confidence if available")


class OCRLine(BaseModel):
    """Line of recognized words with bounding box and center coordinates."""

    text: str
    words: List[OCRWord] = Field(default_factory=list)
    rect: Dict[str, int] = Field(description="Enclosing bounding box for line")
    center: Tuple[int, int] = Field(description="Center point (x, y)")


class OCRResult(BaseModel):
    """Structured result returned from native Windows OCR observation."""

    status: str
    text: str = ""
    lines: List[OCRLine] = Field(default_factory=list)
    word_count: int = 0
    line_count: int = 0
    region: Optional[Tuple[int, int, int, int]] = Field(
        default=None, description="Screen region (left, top, right, bottom) if region OCR was performed"
    )
    hwnd: Optional[int] = Field(default=None, description="Target or active window handle when observed")
    screenshot_path: Optional[str] = Field(default=None, description="Path to screenshot image file if saved")
    error: Optional[str] = None


# Embedded PowerShell WinRT script for offline Windows.Media.Ocr invocation
_PS_OCR_SCRIPT = r"""
param([string]$ImagePath, [string]$LanguageTag)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

try {
    [Reflection.Assembly]::LoadWithPartialName('System.Runtime.WindowsRuntime') | Out-Null
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation.Diagnostics, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null

    $asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
    } | Select-Object -First 1

    function Await-WinRT($asyncOp, $type) {
        $m = $asTaskGeneric.MakeGenericMethod($type)
        $task = $m.Invoke($null, @($asyncOp))
        return $task.GetAwaiter().GetResult()
    }

    if (-not (Test-Path $ImagePath)) {
        @{ status = 'OCR_FAILED'; error = "Image file not found: $ImagePath"; text = ''; lines = @() } | ConvertTo-Json -Compress
        exit 0
    }

    $file = Await-WinRT ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Await-WinRT ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $softwareBitmap = Await-WinRT ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

    if ($LanguageTag) {
        $lang = New-Object Windows.Globalization.Language($LanguageTag)
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
    } else {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    }

    if (-not $engine) {
        @{ status = 'OCR_ENGINE_UNAVAILABLE'; error = 'Windows OCR engine unavailable for user profile languages'; text = ''; lines = @() } | ConvertTo-Json -Compress
        exit 0
    }

    $result = Await-WinRT ($engine.RecognizeAsync($softwareBitmap)) ([Windows.Media.Ocr.OcrResult])

    $linesOut = @()
    foreach ($line in $result.Lines) {
        $wordsOut = @()
        foreach ($word in $line.Words) {
            $wordsOut += @{
                text = $word.Text
                rect = @{
                    x = [int]$word.BoundingRect.X
                    y = [int]$word.BoundingRect.Y
                    width = [int]$word.BoundingRect.Width
                    height = [int]$word.BoundingRect.Height
                }
            }
        }
        $linesOut += @{
            text = $line.Text
            words = $wordsOut
        }
    }

    $status = if ($result.Text -and $result.Text.Trim().Length -gt 0) { 'OCR_SUCCESS_TEXT_FOUND' } else { 'OCR_SUCCESS_NO_TEXT' }
    @{
        status = $status
        text = $result.Text
        lines = $linesOut
    } | ConvertTo-Json -Compress -Depth 6
} catch {
    @{
        status = 'OCR_FAILED'
        error = $_.Exception.Message
        text = ''
        lines = @()
    } | ConvertTo-Json -Compress
}
"""


class WindowsNativeOCR:
    """Windows-native OCR abstraction using Windows.Media.Ocr WinRT engine."""

    def __init__(self, language_tag: Optional[str] = None) -> None:
        self.language_tag = language_tag

    def recognize_image(
        self,
        image_path: Union[str, Path],
        offset: Tuple[int, int] = (0, 0),
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> OCRResult:
        """Run native Windows OCR on an image file and translate coordinates with given offset."""
        img_path = Path(image_path).resolve()
        if not img_path.exists():
            return OCRResult(
                status=OCR_FAILED,
                error=f"Image file does not exist: '{img_path}'",
            )

        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"& {{ {_PS_OCR_SCRIPT} }} -ImagePath \"{str(img_path)}\""
            + (f" -LanguageTag \"{self.language_tag}\"" if self.language_tag else ""),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20.0,
            )
            raw_output = proc.stdout.strip()
            if not raw_output:
                err_msg = proc.stderr.strip() or "Empty output from OCR runner"
                return OCRResult(status=OCR_FAILED, error=err_msg)

            data = json.loads(raw_output)
        except subprocess.TimeoutExpired:
            return OCRResult(status=OCR_FAILED, error="OCR recognition timed out after 20 seconds.")
        except Exception as e:
            return OCRResult(status=OCR_FAILED, error=f"Failed to execute OCR engine: {e}")

        status = data.get("status", OCR_FAILED)
        if status == OCR_FAILED:
            return OCRResult(
                status=OCR_FAILED,
                error=data.get("error", "Unknown OCR failure"),
                screenshot_path=str(img_path),
            )
        if status == OCR_ENGINE_UNAVAILABLE:
            return OCRResult(
                status=OCR_ENGINE_UNAVAILABLE,
                error=data.get("error", "Windows OCR engine unavailable"),
                screenshot_path=str(img_path),
            )

        offset_x, offset_y = offset
        raw_lines = data.get("lines", [])
        parsed_lines: List[OCRLine] = []
        total_words = 0

        for l_item in raw_lines:
            line_text = l_item.get("text", "")
            raw_words = l_item.get("words", [])
            words: List[OCRWord] = []

            for w_item in raw_words:
                w_text = w_item.get("text", "")
                r = w_item.get("rect", {})
                lx = int(r.get("x", 0))
                ly = int(r.get("y", 0))
                lw = int(r.get("width", 0))
                lh = int(r.get("height", 0))

                abs_left = offset_x + lx
                abs_top = offset_y + ly
                abs_right = abs_left + lw
                abs_bottom = abs_top + lh
                center_pt = (abs_left + (lw // 2), abs_top + (lh // 2))

                words.append(
                    OCRWord(
                        text=w_text,
                        rect={
                            "left": abs_left,
                            "top": abs_top,
                            "right": abs_right,
                            "bottom": abs_bottom,
                            "width": lw,
                            "height": lh,
                        },
                        center=center_pt,
                    )
                )

            total_words += len(words)

            if words:
                line_left = min(w.rect["left"] for w in words)
                line_top = min(w.rect["top"] for w in words)
                line_right = max(w.rect["right"] for w in words)
                line_bottom = max(w.rect["bottom"] for w in words)
                line_w = line_right - line_left
                line_h = line_bottom - line_top
                line_center = (line_left + (line_w // 2), line_top + (line_h // 2))
                line_rect = {
                    "left": line_left,
                    "top": line_top,
                    "right": line_right,
                    "bottom": line_bottom,
                    "width": line_w,
                    "height": line_h,
                }
            else:
                line_rect = {"left": offset_x, "top": offset_y, "right": offset_x, "bottom": offset_y, "width": 0, "height": 0}
                line_center = (offset_x, offset_y)

            parsed_lines.append(
                OCRLine(
                    text=line_text,
                    words=words,
                    rect=line_rect,
                    center=line_center,
                )
            )

        full_text = data.get("text", "").strip()
        final_status = OCR_SUCCESS_TEXT_FOUND if full_text else OCR_SUCCESS_NO_TEXT

        return OCRResult(
            status=final_status,
            text=full_text,
            lines=parsed_lines,
            word_count=total_words,
            line_count=len(parsed_lines),
            region=region,
            screenshot_path=str(img_path),
        )

    def _attach_interactive_desktop(self) -> Optional[int]:
        """Attach current thread to the default interactive desktop station."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            cur_desk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
            if cur_desk:
                return None
            hdesk = user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                if user32.SetThreadDesktop(hdesk):
                    return hdesk
                else:
                    user32.CloseDesktop(hdesk)
        except Exception:
            pass
        return None

    def _detach_interactive_desktop(self, hdesk: Optional[int]) -> None:
        pass

    def capture_window(self, hwnd: int, save_path: Optional[Path] = None) -> Tuple[Optional[Any], Optional[Dict[str, int]]]:
        """Capture rendered window surface using Win32 PrintWindow (PW_RENDERFULLCONTENT)."""
        import ctypes
        from ctypes import wintypes
        from PIL import Image

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None, None

        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w <= 0 or h <= 0:
            return None, None

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        gdi32.SelectObject(hdc_mem, hbm)

        # PW_RENDERFULLCONTENT = 2
        ok = user32.PrintWindow(hwnd, hdc_mem, 2)
        if not ok:
            ok = user32.PrintWindow(hwnd, hdc_mem, 0)

        if not ok:
            gdi32.DeleteObject(hbm)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
            return None, None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)
        img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        rect_dict = {
            "left": int(rect.left),
            "top": int(rect.top),
            "right": int(rect.right),
            "bottom": int(rect.bottom),
            "width": w,
            "height": h,
        }
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p)

        return img, rect_dict

    def recognize_screen(self, save_path: Optional[Path] = None, hwnd: Optional[int] = None) -> OCRResult:
        """Capture the full primary display (or targeted window) and perform OCR recognition."""
        from PIL import Image, ImageGrab

        temp_created = False
        if save_path is None:
            fd, tmp_file = tempfile.mkstemp(suffix=".png", prefix="ocr_screen_")
            os.close(fd)
            target = Path(tmp_file)
            temp_created = True
        else:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)

        if hwnd is not None:
            win_img, win_rect = self.capture_window(int(hwnd), save_path=target)
            if win_img and win_rect:
                try:
                    return self.recognize_image(
                        target,
                        offset=(win_rect["left"], win_rect["top"]),
                        region=(win_rect["left"], win_rect["top"], win_rect["right"], win_rect["bottom"]),
                    )
                finally:
                    if temp_created and target.exists():
                        try:
                            target.unlink()
                        except Exception:
                            pass

        hdesk = self._attach_interactive_desktop()
        try:
            try:
                img = ImageGrab.grab()
            except Exception:
                import ctypes
                w = ctypes.windll.user32.GetSystemMetrics(0) or 1920
                h = ctypes.windll.user32.GetSystemMetrics(1) or 1080
                img = Image.new("RGB", (w, h), color=(255, 255, 255))
            img.save(target)
            result = self.recognize_image(target, offset=(0, 0), region=(0, 0, img.width, img.height))
            return result
        finally:
            self._detach_interactive_desktop(hdesk)
            if temp_created and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass

    def recognize_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        save_path: Optional[Path] = None,
        hwnd: Optional[int] = None,
    ) -> OCRResult:
        """Capture a bounded desktop screen region and perform OCR with absolute coordinate translation."""
        from PIL import Image, ImageGrab

        if width <= 0 or height <= 0:
            return OCRResult(
                status=OCR_FAILED,
                error=f"Invalid region dimensions: width={width}, height={height} (must be > 0).",
            )
        if x < 0 or y < 0:
            return OCRResult(
                status=OCR_FAILED,
                error=f"Invalid region coordinates: x={x}, y={y} (coordinates cannot be negative).",
            )

        bbox = (int(x), int(y), int(x) + int(width), int(y) + int(height))
        temp_created = False
        if save_path is None:
            fd, tmp_file = tempfile.mkstemp(suffix=".png", prefix="ocr_region_")
            os.close(fd)
            target = Path(tmp_file)
            temp_created = True
        else:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)

        if hwnd is not None:
            win_img, win_rect = self.capture_window(int(hwnd))
            if win_img and win_rect:
                try:
                    lx = max(0, int(x) - win_rect["left"])
                    ly = max(0, int(y) - win_rect["top"])
                    lw = min(win_img.width - lx, int(width))
                    lh = min(win_img.height - ly, int(height))
                    if lw > 0 and lh > 0:
                        cropped = win_img.crop((lx, ly, lx + lw, ly + lh))
                    else:
                        cropped = win_img
                    cropped.save(target)
                    return self.recognize_image(
                        target,
                        offset=(win_rect["left"] + lx, win_rect["top"] + ly),
                        region=bbox,
                    )
                finally:
                    if temp_created and target.exists():
                        try:
                            target.unlink()
                        except Exception:
                            pass

        hdesk = self._attach_interactive_desktop()
        try:
            try:
                img = ImageGrab.grab(bbox=bbox)
            except Exception:
                img = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
            img.save(target)
            result = self.recognize_image(
                target,
                offset=(int(x), int(y)),
                region=bbox,
            )
            return result
        finally:
            self._detach_interactive_desktop(hdesk)
            if temp_created and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
