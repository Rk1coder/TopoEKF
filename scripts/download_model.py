from ultralytics import YOLO


def main() -> None:
    YOLO("yolo12n.pt")


if __name__ == "__main__":
    main()
