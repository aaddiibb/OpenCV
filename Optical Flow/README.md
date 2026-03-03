# Optical Flow Visualization using HSV Color Space

This project demonstrates **Optical Flow** computation using two popular algorithms:
1. **Farneback Algorithm** (Dense Optical Flow)
2. **Lucas-Kanade Algorithm** (Sparse Optical Flow)

Both are visualized using the HSV (Hue-Saturation-Value) color space for intuitive motion understanding.

---

## What is Optical Flow?

Optical flow is the pattern of apparent motion of objects between consecutive video frames. Think of it as answering the question: **"Where did each pixel move?"**

### The Basic Idea (Step by Step)

```
Frame 1                    Frame 2
┌─────────────┐           ┌─────────────┐
│      ●      │    →      │         ●   │
│             │           │             │
└─────────────┘           └─────────────┘

The dot moved from left to right!
```

**For every pixel, we compute a motion vector: `(dx, dy)`**

- `dx` = how much the pixel moved horizontally
- `dy` = how much the pixel moved vertically

---

## From Motion Vectors to HSV: The Magic

Once we have the motion vector `(dx, dy)` for each pixel, we compute two things:

### 1. Direction (Angle)
```
angle = atan2(dy, dx)
```
This tells us **which direction** the pixel moved (up, down, left, right, diagonal, etc.)

### 2. Speed (Magnitude)
```
magnitude = √(dx² + dy²)
```
This tells us **how fast** the pixel moved (using Pythagorean theorem!)

### Visual Example:
```
     dy
      ↑
      │   /  ← motion vector (dx, dy)
      │  /
      │ /  angle (direction)
      │/________→ dx
      
      magnitude = length of the arrow
```

---

## Storing Motion in HSV Image

Here's where the magic happens! We map motion to colors:

```
┌─────────────────────────────────────────────────────────┐
│                    HSV IMAGE                            │
├─────────────────────────────────────────────────────────┤
│  H (Hue / Color)      = Direction of motion             │
│  S (Saturation)       = Kept HIGH (255) for vivid colors│
│  V (Value/Brightness) = Speed of motion                 │
└─────────────────────────────────────────────────────────┘
```

### Why This Works So Well:

| Motion Property | HSV Channel | Visual Result |
|-----------------|-------------|---------------|
| **Direction** | Hue (H) | Different colors for different directions |
| **Speed** | Value (V) | Brighter = faster, Darker = slower |
| **Saturation** | S = 255 | Always vivid, easy to see |

### Color Wheel for Direction:

```
                    UP (Magenta/Pink)
                         ↑
                         │
    UP-LEFT (Purple) ←───┼───→ UP-RIGHT (Red)
                         │
    LEFT (Blue) ←────────┼────────→ RIGHT (Red)
                         │
  DOWN-LEFT (Cyan) ←─────┼─────→ DOWN-RIGHT (Yellow)
                         │
                         ↓
                   DOWN (Green)
```

---

## Two Types of Optical Flow

### 1. Dense Optical Flow (Farneback)
- Computes motion for **EVERY pixel** in the frame
- Produces a complete motion map
- More computationally intensive
- Best for: Understanding overall scene motion

### 2. Sparse Optical Flow (Lucas-Kanade)
- Tracks only **specific points** (usually corners)
- Faster and more efficient
- Best for: Object tracking, feature tracking

```
Dense (Farneback)              Sparse (Lucas-Kanade)
┌────────────────┐             ┌────────────────┐
│→→→→→→→→→→→→→→→→│             │    ●→          │
│→→→→→→→→→→→→→→→→│             │         ●→     │
│→→→→→→→→→→→→→→→→│             │  ●→            │
│→→→→→→→→→→→→→→→→│             │        ●→      │
└────────────────┘             └────────────────┘
  Every pixel tracked           Only key points tracked
```

---

# PART 1: Farneback Algorithm (Dense Optical Flow)

Farneback computes optical flow for **every pixel** using polynomial expansion.

## Code Breakdown

### Step 1: Import Libraries

```python
import cv2
import numpy as np
```

| Library | Purpose |
|---------|---------|
| `cv2` | OpenCV library for computer vision |
| `numpy` | Numerical operations on arrays |

---

### Step 2: Open the Video

```python
capture = cv2.VideoCapture("input.mp4")
if not capture.isOpened():
    raise RuntimeError("Cannot open video file")
```

**What it does:**
- Opens the video file for reading
- Checks if the video opened successfully
- Raises an error if it fails (better than crashing later!)

---

### Step 3: Read the First Frame

```python
ret, frame1 = capture.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
```

**What it does:**
- Reads the first frame from the video
- `ret` = True if successful, False if failed
- Converts to **grayscale** because optical flow works on brightness, not color

**Why grayscale?**
1. Simpler: 1 channel instead of 3
2. Optical flow tracks **intensity changes**, not color changes
3. Faster computation

---

### Step 4: Create the HSV Canvas

```python
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255
```

**What it does:**
- Creates a black image same size as our video frame
- Sets Saturation to 255 (maximum) for vivid colors

```
hsv_mask has 3 channels:
┌─────────────────────────────────────┐
│ Index 0: Hue        → Will store direction │
│ Index 1: Saturation → Set to 255 (vivid)   │
│ Index 2: Value      → Will store speed     │
└─────────────────────────────────────┘
```

---

### Step 5: Main Loop - Read Each Frame

```python
while True:
    ret, frame2 = capture.read()
    if not ret:
        break
```

**What it does:**
- Reads frames one by one
- Stops when video ends

---

### 6. Grayscale Conversion

```python
next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
```

Converts the current frame to grayscale for optical flow computation.

---

### 7. Farneback Optical Flow Calculation

```python
flow = cv2.calcOpticalFlowFarneback(
    prvs, next_gray, None,
    0.5, 3, 15, 3, 5, 1.2, 0
)
```

**This is the core of the algorithm.** It computes dense optical flow using Gunnar Farneback's algorithm.

**Parameters Explained:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `prvs` | - | Previous grayscale frame |
| `next_gray` | - | Current grayscale frame |
| `None` | - | Output flow array (None = create new) |
| `pyr_scale` | 0.5 | Image pyramid scale (0.5 = each level is half the size) |
| `levels` | 3 | Number of pyramid levels |
| `winsize` | 15 | Averaging window size (larger = more robust, less detail) |
| `iterations` | 3 | Number of iterations at each pyramid level |
| `poly_n` | 5 | Size of pixel neighborhood for polynomial expansion |
| `poly_sigma` | 1.2 | Gaussian standard deviation for polynomial expansion |
| `flags` | 0 | Operation flags |

**Output:**
- `flow` is a 2-channel array of shape `(height, width, 2)`
- `flow[..., 0]` = Horizontal displacement (dx)
- `flow[..., 1]` = Vertical displacement (dy)

---

### 8. Converting to Polar Coordinates

```python
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

**What it does:**
Converts Cartesian coordinates (dx, dy) to Polar coordinates (magnitude, angle).

| Output | Meaning |
|--------|---------|
| `mag` | **Speed** of motion (how fast pixels moved) |
| `ang` | **Direction** of motion (in radians) |

**Mathematical relationship:**
- `magnitude = √(dx² + dy²)`
- `angle = atan2(dy, dx)`

---

### 9. HSV Encoding

```python
hsv_mask[..., 0] = ang * 180 / np.pi / 2
hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
```

**Hue Channel (Direction):**
- `ang * 180 / np.pi` - Converts radians to degrees (0-360°)
- `/ 2` - Scales to OpenCV's Hue range (0-180)
- Result: Different colors represent different motion directions

**Value Channel (Magnitude):**
- `cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)` - Normalizes magnitude to 0-255 range
- Brighter pixels = Faster motion
- Darker pixels = Slower or no motion

---

### 10. Color Space Conversion & Display

```python
flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
cv2.imshow("Optical Flow (HSV)", flow_bgr)
```

**What it does:**
- Converts HSV image to BGR for display (OpenCV uses BGR by default)
- Shows the visualization in a window

---

### 11. Keyboard Controls

```python
k = cv2.waitKey(20) & 0xFF
if k == ord('e'):
    break
elif k == ord('s'):
    cv2.imwrite("frame.png", frame2)
    cv2.imwrite("optical_flow.png", flow_bgr)
```

| Key | Action |
|-----|--------|
| `e` | **Exit** the program |
| `s` | **Save** current frame and optical flow image |

The `& 0xFF` is a bitwise AND operation that keeps only the last 8 bits (necessary for cross-platform compatibility).

---

### 12. Frame Update

```python
prvs = next_gray
```

Updates the "previous frame" to the current frame for the next iteration. This creates the sliding window effect where each frame is compared to its predecessor.

---

### 13. Cleanup

```python
capture.release()
cv2.destroyAllWindows()
```

| Function | Purpose |
|----------|---------|
| `capture.release()` | Releases the video file handle |
| `cv2.destroyAllWindows()` | Closes all OpenCV windows |

---

## Visual Output Interpretation

The output visualization uses colors to show motion:

| Color | Motion Direction |
|-------|-----------------|
| 🔴 Red | Right |
| 🟡 Yellow | Down-Right |
| 🟢 Green | Down |
| 🔵 Cyan | Down-Left |
| 🔵 Blue | Left |
| 🟣 Magenta | Up-Left |
| 🔴 Red/Pink | Up |

**Brightness indicates speed:**
- **Bright** = Fast motion
- **Dark** = Slow or no motion

---

## Requirements

```
opencv-python
numpy
```

Install with:
```bash
pip install opencv-python numpy
```

---

## Usage

1. Place your video file as `input.mp4` in the same directory
2. Run the notebook cell
3. Press `s` to save frames, `e` to exit

---

## Applications

- **Video Stabilization** - Detect unwanted camera motion
- **Object Tracking** - Follow moving objects
- **Action Recognition** - Understand motion patterns
- **Autonomous Vehicles** - Detect moving obstacles
- **Video Compression** - Predict frame content based on motion

















