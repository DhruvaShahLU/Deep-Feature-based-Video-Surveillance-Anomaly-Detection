# Deep Feature Based Video_Surveillance Anomaly Detection



<img src="https://user-images.githubusercontent.com/121331076/233763113-dec9bd02-9cab-4f8f-9ecc-ab92a022ccf8.png">

Detecting anomalies in video surveillance is a chal-
lenging task that requires distinguishing between normal and
abnormal behaviour. Video surveillance systems not only face
challenges in identifying and monitoring unusual human actions
but also in differentiating normal from anomalous actions due
to a large amount of data in video format. In this study, we
propose an intelligent video surveillance system that utilizes deep
feature-based anomaly detection to identify anomalous events in
a video stream. Our approach uses a two-stream deep learning
model with I3D as the feature extraction component, which has
demonstrated effectiveness in action recognition and detection
tasks. We evaluate our proposed system on the UCF Crime
dataset, consisting of videos of normal and abnormal occurrences,
and achieve an AUC of 87.52%. Our results demonstrate the
efficacy of the proposed method in identifying anomalous events
and show significant improvement over state-of-the-art methods


The path to the extracted features dataset:-
https://drive.google.com/drive/folders/1pIUMAkYJjX0SWTmdDxATi_sSa0W26oSi?usp=share_link

Download the above data link and extract under your $DATA_ROOT_DIR. /workspace/UCF-Crime/all_rgbs


**Directory tree**
    UCF-Crime/<br>
        ../all_rgbs<br>
            ../.npy<br>
        ./all_flows<br>
            ../.npy<br>
        train_anomaly.txt<br>
        train_normal.txt<br>
        test_anomaly.txt<br>
        test_normal.txt<br>

       
**train-test script**

```python main.py```

**Results**
To replicate the results please load best_model.pth
<img src="">

**For Visualization**
```python video2frame.py --n filename.mp4```
```python vis.py --n filename```
<img src="">
<img src="">
