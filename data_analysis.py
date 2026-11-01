

import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import Counter

st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 AI Smart Traffic Monitoring System")
st.write("Vehicle Detection, Counting and Traffic Analysis")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Save uploaded video
    with open("traffic_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("traffic_video.mp4")

    st.subheader("Traffic Video Analysis")

    video_display = st.empty()

    vehicle_count = Counter()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id in vehicle_classes:

                    vehicle_name = vehicle_classes[class_id]
                    vehicle_count[vehicle_name] += 1

        annotated_frame = results[0].plot()

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        video_display.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    st.success("Traffic Analysis Completed!")

    # Display results
    st.subheader("Vehicle Detection Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cars", vehicle_count["Car"])
    col2.metric("Motorcycles", vehicle_count["Motorcycle"])
    col3.metric("Buses", vehicle_count["Bus"])
    col4.metric("Trucks", vehicle_count["Truck"])

    total = sum(vehicle_count.values())

    st.metric("Total Vehicle Detections", total)

    # DataFrame
    df = pd.DataFrame(
        list(vehicle_count.items()),
        columns=["Vehicle Type", "Count"]
    )

    st.subheader("Vehicle Statistics")

    st.dataframe(df, use_container_width=True)

    # Graph
    fig, ax = plt.subplots()

    ax.bar(df["Vehicle Type"], df["Count"])

    ax.set_title("Vehicle Detection Analysis")
    ax.set_xlabel("Vehicle Type")
    ax.set_ylabel("Detection Count")

    st.pyplot(fig)

    # Download report
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Traffic Report",
        csv,
        "traffic_report.csv",
        "text/csv"
    )

else:
    st.info("Please upload a traffic video to start analysis.")
