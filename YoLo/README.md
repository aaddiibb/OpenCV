# Suspicious Activity Detection with YOLOv8-Pose

A beginner-friendly guide to understanding how this project detects suspicious activities in videos using artificial intelligence.

---

## 📚 Table of Contents
1. [What is YOLO?](#what-is-yolo)
2. [How This Code Works](#how-this-code-works)
3. [Step-by-Step Detection Process](#step-by-step-detection-process)
4. [The Suspicious Activity Rules](#the-suspicious-activity-rules)
5. [How to Use](#how-to-use)
6. [Understanding the Code](#understanding-the-code)

---

## What is YOLO?

### Simple Explanation
**YOLO** = **"You Only Look Once"** - it's an AI model that can see pictures and identify what's in them **super fast**.

Instead of looking at a picture piece by piece (which is slow), YOLO looks at the **entire picture at once** and instantly says:
- "I see a person here"
- "I see a car there"
- "I see where their head, arms, and legs are"

Think of it like how **your eyes work** - you don't analyze each part of the room individually; you see everything at once!

### What Does YOLOv8-Pose Do?

Our code uses a special version called **YOLOv8-Pose**. This version doesn't just find people - it also finds their **body parts (keypoints)**.

**Keypoints** = the joints and important points on a person's body:
- Head
- Shoulders
- Elbows
- Wrists (hands)
- Hips
- Knees
- Ankles

The AI can track all 17 different keypoints on a person! It works like having a skeleton overlay on each person in the video.

---

## How This Code Works

### The Big Picture (3 Simple Steps)

```
1. WATCH VIDEO → 2. FIND PEOPLE & THEIR POSES → 3. CHECK IF THEY'RE DOING SOMETHING SUSPICIOUS
```

That's it! Let me explain each step:

---

## Step-by-Step Detection Process

### **STEP 1: Read Video Frame by Frame**

```python
cap = cv2.VideoCapture("input.mp4")  # Open the video
```

Think of a video as a **stack of pictures** shown really fast:
- One picture = one "frame"
- A 30-second video = thousands of frames
- Your code looks at each picture one at a time

### **STEP 2: Detect People and Their Poses**

```python
res = model.predict(frame, conf=0.35)
```

For **every frame**, the AI model does this:

1. **Scans the entire picture** to find people
2. **Draws a box** around each person
3. **Finds 17 body keypoints** (joints) on each person
4. **Gives each person a "confidence score"** (how sure the AI is)
   - 1.0 = 100% sure this is a person
   - 0.5 = 50% sure
   - Our code only trusts detections above **0.35 (35% confidence)**

### **STEP 3: Check for Suspicious Behavior**

```python
suspicious, label = detect_suspicious_activity(kpts_i, H)
```

This is where the **magic happens**! The code looks at the keypoints and checks **if the person is doing something unusual**.

---

## The Suspicious Activity Rules

The code checks for **5 different suspicious poses**:

### **Rule 1️⃣: Hands Up** 🙌
**What it detects:** Both hands above the head
```
Nose position = Y-coordinate of the head
Left Wrist position = Y-coordinate of left hand
Right Wrist position = Y-coordinate of right hand

SUSPICIOUS IF: Both wrists are above the nose
```
**Why it matters:** Could indicate robbery, surrender, or distress

---

### **Rule 2️⃣: Crouching** 🤫
**What it detects:** Someone bending their knees a lot (like hiding or preparing to run)
```
Formula:
- Measure torso length = distance from hips to shoulders
- Measure thigh length = distance from knees to hips

SUSPICIOUS IF: Thigh length < 40% of torso length
```
**Why it matters:** Could indicate someone hiding, preparing to attack, or stealing

---

### **Rule 3️⃣: Lying Down** 😴
**What it detects:** Person is horizontal (lying flat)
```
Check if shoulders, hips, and knees are almost at the same height (very close Y values)

SUSPICIOUS IF: All three are within a small distance of each other
```
**Why it matters:** Person might be injured, unconscious, or climbing

---

### **Rule 4️⃣: Crawling** 🐕
**What it detects:** Person moving on hands and knees
```
Check if:
1. Knees are at the same level as hips (nearly touching vertically)
2. Shoulders are BELOW the hips

SUSPICIOUS IF: Both conditions are true
```
**Why it matters:** Could indicate someone trying to hide or escape

---

### **Rule 5️⃣: Fall** 📉
**What it detects:** Person's head is below their hips (they've fallen)
```
SUSPICIOUS IF: Nose position (head) is LOWER than hip position
```
**Why it matters:** Person might be injured and needs help

---

## How to Use

### **Installation** 📦

```bash
# Install required packages
pip install ultralytics opencv-python

# The first time you run:
# Download YOLO model (automatically)
```

### **Running the Code** ▶️

1. Place your video file in the same folder as the code
2. In the code, change the filename:
   ```python
   cfg = Config(
       input_video="YOUR_VIDEO.mp4",  # ← Change this
       output_video="output_suspicious.mp4"
   )
   ```

3. Run the code (it will process frame by frame)
4. Check the output file: `output_suspicious.mp4`

### **Output Video** 📹

The output video will show:
- **Green boxes** = Normal person
- **Red boxes** = Suspicious activity detected
- **Labels** = What suspicious activity was found

---

## Understanding the Code

### **Configuration (Settings)**

```python
@dataclass
class Config:
    pose_model: str = "yolov8n-pose.pt"  # AI model to use
    conf: float = 0.35                    # How confident AI must be (35%)
    draw_keypoints: bool = True           # Show body joint dots
```

### **The Detection Function (Main Logic)**

```python
def detect_suspicious_activity(kpts_xy, frame_height):
    # kpts_xy = the 17 keypoints (body joints) as X,Y coordinates
    # frame_height = height of the video frame (for scaling)
```

This function:
1. **Extracts key positions** (nose, shoulders, hips, knees, wrists)
2. **Calculates distances** between body parts
3. **Checks all 5 rules** in order
4. **Returns** True/False and a label describing what it found

### **Video Processing Loop**

```python
while True:
    ok, frame = cap.read()        # Get next frame
    if not ok:
        break                      # No more frames
    
    res = model.predict(frame)     # Find people
    
    for each person detected:
        check if suspicious        # Apply rules
        draw box + label           # Draw on frame
    
    writer.write(frame)            # Save to output video
```

---

## 🧠 How the AI "Thinks"

1. **Training**: Before we used it, someone trained YOLO on millions of pictures of people
2. **Pattern Recognition**: YOLO learned to recognize: "This pattern looks like a person's shoulder"
3. **Prediction**: When we show it a new picture, it uses what it learned

### Why YOLOv8-Pose is Special:
- **Fast**: Processes 30-60 frames per second
- **Accurate**: Detects keypoints even if person is partially hidden
- **Smart**: Understands body structure (knee below hip, etc.)

## 📝 Summary

This code is like a **smart security camera** that:
1. Watches a video frame by frame ⏱️
2. Finds every person and their body parts 👥
3. Checks if they're doing something suspicious 🚨
4. Marks them with red boxes in the output video 📹


