# IoT-Intrusion-Detection-using-Machine-Learning
This work was done as part of a team to develop a solution for detecting network intrusions
This work uses a customized the AEGEAN Wifi dataset available here: http://icsdweb.aegean.gr/awid/
The problem was a balanced binary classification one; several pre-processing, feature selection/generation steps and machine learning models were considered.


# Project/Result Summary
Scoring metrics are used to evaluate the performance of the various ML models applied to any
dataset. For this work the Classification accuracy (CA), Detection rate (DR) (or True Positive Rate,
TPR), False alarm rate (FAR), “Receiving Operating Characteristic” (ROC) will be used to evaluate
and compare the various models. The results are also benchmarked against what is understood to be
the best results for this dataset and impersonation attacks classification – a DR of 99.918%, Accuracy
of 99.97%, FPR of 0.012% [22]. 

The best classifier result for this project (ENS XG + ADA) has a detection rate of 100% for
impersonation attacks and a false positive rate of 5%. The detection rate exceeds the highest level
seen in literature [22]. However, our FPR is 5% which is significantly higher than the 0.012% also
reported by [22].

Full report available on request

[22]  M. E. Aminanto, R. Choi, H. C. Tanuwidjaja, P. D. Yoo and K. Kim, “Deep abstraction and
weighted feature selection for Wi-Fi impersonation detection,” IEEE Transactions on
Information Forensics and Security, vol. 13, no. 3, pp. 621-636, 12 10 2017. 
