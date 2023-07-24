# Annotation with CVAT
To start:
```python
cd cvat
docker-compose up -d
docker exec -t cvat_server python manage.py health_check
```

Stop all containers
The command below stops and removes containers and networks created by up.
```python
docker compose down
```

## Facebook SAM for semi-supervised annotation
1. Prepare the "magic wand" - CVAT's embedded tool for AI
    * https://opencv.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/
    * https://www.cvat.ai/post/facebook-segment-anything-model-in-cvat
    * https://github.com/opencv/cvat/issues/6041
    * https://github.com/opencv/cvat/issues/4736
2. Increase timeouts and decrease workers num in case of weak hardware in the function code /serverless/pytorch/facebookresearch/sam/nuclio/function.yaml
3. Updating bash on MacOs to be able to launch deploy_cpu.sh: https://itnext.io/upgrading-bash-on-macos-7138bd1066ba
4. From cvat folder
```sh
bash ./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio/
```
5. CVAT address: http://localhost:8080/
6. nuclio address: http://localhost:8070/

# Resources:
1. https://telegra.ph/Razmechaem-dannye-Bystro-nedorogo-12-27

# Project roadmap:
1. Define labels of interest. It was decided to take doll brands as the main goal of  the task. The specificity of the task is that they have various of different accecories so that the same doll may be in very different clother and with different items weared on. Here we don't take accecories into account and just ignore them during the annotation process. However, clothes which are close to body are important and are annotated together with the dolls themselves.
2. It was decided to annotate the project in the CVAT with an optional help of SAM to speed up the annotation and increase the quality of it. About 50 images is enough to check the pipeline and formats.
3. Try zero-shot: https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/
    1. Evaluate on existing bboxes
        1. any doll is the target
        2. every original class (barbie, bratz) is the goal
4. Few-shot?
5. Consider distilling (probably, with fine-tuning) DINO to YOLO: https://github.com/autodistill/autodistill
6. fine-tune YOLO v5 or v7 on bbox
7. Optimize for mobile
