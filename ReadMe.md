> Các tài liệu tham khảo:
>
> Các module được define rõ ràng nhất là ở mm detection: [https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5]()
>
> Cách define dataloader: [https://jovian.com/zachwad/boat-pytorch-yolov5#C15]()
>
> Cách define CSSA module: [https://github.com/artrela/mulitmodal-cssa/blob/main/model-train.ipynb]()
>
> Tham khảo tool train của YOLOX: [https://github.com/Megvii-BaseDetection/YOLOX]()
>
> Link datasets: [https://drive.google.com/file/d/1BTlpXGTtcE9nW0p6PWF1-JuxJCkD6fnv/view?usp=sharing](https://drive.google.com/file/d/1BTlpXGTtcE9nW0p6PWF1-JuxJCkD6fnv/view?usp=sharing)

> **Các step cần thực hiện:**
>
> **Step 1:** Nghiên cứu phát triển model kết hợp khối CSSA thỏa mãn 2 đầu vào là ảnh RGB và ảnh IR
>
> **Step 2:** Nghiên cứu phát triển dataloader, data augmentation cho data
>
> **Step 3:** Phát triển tools training và eval cho new model

* [X] Nhiệm vụ 1: Nghiên cứu, phát triển model 2 đầu vào based on trên YOLO
* [X] Nhiệm vụ 2: Thiết kế dataloader cho model
* [ ] Nhiệm vụ 3: Viết tool train cho model
