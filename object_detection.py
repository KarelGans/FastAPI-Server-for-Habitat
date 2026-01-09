from typing import List, Dict, Union
from PIL import Image
import supervision as sv  # type: ignore
from ultralytics import YOLOE


class ObjectDetector:
    """
    A standalone class for YOLOe object detection.
    Provides both annotated images and bounding boxes.
    """

    def __init__(self, checkpoint: str = "yoloe-v8l-seg.pt", device: str = "cpu"):
        """
        Initialize the YOLOe model.

        Args:
            checkpoint (str): Path or ID of YOLOe checkpoint.
            device (str): Device to run inference on, e.g., "cpu" or "cuda:0".
        """
        self.model = YOLOE(checkpoint)
        self.model.to(device)
        self.device = device

    def ObjectDetection(
        self,
        img_source: Union[str, Image.Image],
        names: List[str] = [],
        output: str | None = None
    ) -> Dict:
        """
        Run detection on the image, annotate it, and return bounding boxes.

        Args:
            img_source (str | PIL.Image): Path to image or PIL Image object.
            names (List[str]): List of object names to detect.
            output (str | None): If provided, saves the annotated image to this path.

        Returns:
            Dict: {
                "image": PIL.Image (annotated),
                "objectCoordinates": List[Dict[str, float]],
                "hasObject": bool
            }
        """
        # Load image
        if isinstance(img_source, str):
            image = Image.open(img_source).convert("RGB")
        else:
            image = img_source.convert("RGB")

        # Set class names for text-guided detection
        self.model.set_classes(names, self.model.get_text_pe(names))

        # Run inference
        results = self.model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # Annotation parameters
        resolution_wh = image.size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

        # Prepare labels and coordinates
        object_coords: List[Dict[str, float]] = []
        labels = []
        for class_name, conf, box in zip(detections["class_name"], detections.confidence, detections.xyxy):
            labels.append(f"{class_name} {conf:.2f}")
            object_coords.append({
                "x_min": float(box[0]),
                "y_min": float(box[1]),
                "x_max": float(box[2]),
                "y_max": float(box[3])
            })

        # Annotate image
        annotated_image = image.copy()
        annotated_image = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.4
        ).annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.BoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=thickness
        ).annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_scale=text_scale,
            smart_position=True
        ).annotate(scene=annotated_image, detections=detections, labels=labels)

        # Save image if requested
        if output:
            annotated_image.save(output)

        return {
            "image": annotated_image,
            "objectCoordinates": object_coords,
            "hasObject": len(object_coords) > 0
        }
    
if __name__ == "__main__":
# Example usage for local testing

    IMAGE_PATH = "annotated_img_logs/wardrobe_false_1.jpg"          # path to a test image
    OBJECT_NAMES = ["Wardrobe"]        # objects to detect
    OUTPUT_PATH = "annotated2.jpg"    # output image

    detector = ObjectDetector(
        checkpoint="yoloe-v8l-seg.pt",
        device="cuda:0"  # change to "cuda:0" if GPU is available
    )

    result = detector.ObjectDetection(
        img_source=IMAGE_PATH,
        names=OBJECT_NAMES,
        output=OUTPUT_PATH
    )

    print("Detection completed")
    print("Has object:", result["hasObject"])
    print("Object coordinates:")
    for coord in result["objectCoordinates"]:
        print(coord)


# import os
# from PIL import Image
# import supervision as sv #type: ignore
# from ultralytics import YOLOE

# def DetectObject(
#     img_source: str,
#     checkpoint: str = "yoloe-v8l-seg.pt",
#     device: str = "cpu",
#     names: list[str] = ["person"],
#     output: str | None = None):
#     # Load image
#     image = Image.open(img_source).convert("RGB")

#     # Load YOLOe model
#     model = YOLOE(checkpoint)
#     model.to(device)

#     # Set class names and embeddings for text-guided detection
#     model.set_classes(names, model.get_text_pe(names))

#     # Run inference
#     results = model.predict(image, verbose=False)

#     # Convert to Supervision Detections
#     detections = sv.Detections.from_ultralytics(results[0])

#     # Annotation parameters
#     resolution_wh = image.size
#     thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
#     text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

#     # Prepare labels
#     labels = [
#         f"{class_name} {confidence:.2f}"
#         for class_name, confidence in zip(detections["class_name"], detections.confidence)
#     ]

#     # Annotate image
#     annotated_image = image.copy()
#     annotated_image = sv.MaskAnnotator(
#         color_lookup=sv.ColorLookup.INDEX,
#         opacity=0.4
#     ).annotate(scene=annotated_image, detections=detections)
#     annotated_image = sv.BoxAnnotator(
#         color_lookup=sv.ColorLookup.INDEX,
#         thickness=thickness
#     ).annotate(scene=annotated_image, detections=detections)
#     annotated_image = sv.LabelAnnotator(
#         color_lookup=sv.ColorLookup.INDEX,
#         text_scale=text_scale,
#         smart_position=True
#     ).annotate(scene=annotated_image, detections=detections, labels=labels)

#     # Determine output path
#     if output is None:
#         base, ext = os.path.splitext(img_source)
#         output = f"{base}-output{ext}"

#     # Save annotated image
#     annotated_image.save(output)
#     print(f"Annotated image saved to: {output}")

#     return output