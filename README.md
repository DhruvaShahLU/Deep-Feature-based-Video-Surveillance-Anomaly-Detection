# Deep Feature Based Video_Surveillance Anomaly Detection



<img src="https://user-images.githubusercontent.com/121331076/233763113-dec9bd02-9cab-4f8f-9ecc-ab92a022ccf8.png">

The path to the extracted features dataset:-
https://drive.google.com/drive/folders/1pIUMAkYJjX0SWTmdDxATi_sSa0W26oSi?usp=share_link


#Directory tree
       UCF-Crime/ 
           ../all_rgbs
               ../~.npy
           ../all_flows
               ../~.npy
       train_anomaly.txt
       train_normal.txt
       test_anomaly.txt
       test_normal.txt

       
For Visualization

python video2frame.py --n filename.mp4
python vis.py --n filename

train-test script

python main.py
