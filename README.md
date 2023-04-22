# Deep Feature Based Video_Surveillance Anomaly Detection



<img src="https://user-images.githubusercontent.com/121331076/233763113-dec9bd02-9cab-4f8f-9ecc-ab92a022ccf8.png">

One of the most difficult difficulties in video surveillance is the ability to distinguish and monitor unusual human actions.
Video surveillance faces challenges in identifying and monitoring unusual human actions, given the large quantity of data available in video format. An intelligent video surveillance system that can detect abnormal actions is proposed in this study. The system learns from a training video collection that includes normal and abnormal occurrences, using a two-stream deep learning model with I3D as the feature extraction part and the UCF Crime dataset. The proposed method achieves an AUC of 87.52\% and demonstrates the effectiveness of the I3D 2stream architecture for video anomaly detection. Evaluations using the UCF Crime dataset show a significant improvement in action recognition compared to state-of-the-art methods.



The path to the extracted features dataset:-
https://drive.google.com/drive/folders/1pIUMAkYJjX0SWTmdDxATi_sSa0W26oSi?usp=share_link


**-Directory tree**
        -UCF-Crime/ 
            -../all_rgbs
                -../~.npy
            -../all_flows
                -../~.npy
        -train_anomaly.txt
        -train_normal.txt
        -test_anomaly.txt
        -test_normal.txt

       
**For Visualization**
```python video2frame.py --n filename.mp4```
```python vis.py --n filename```

**train-test script**

```python main.py```
