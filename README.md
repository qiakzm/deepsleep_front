# **2023 데이터청년캠퍼스 : Deep Sleep**

**Deep Sleep** 은 스마트워치 데이터를 이용해 수면 무호흡증을 예측하는 서비스입니다. AHI 지수를 예측하여 수면무호흡증의 중증도를 알려줍니다.
<br><br>
![DeepSleepDemo_Main1](https://github.com/alacori/deepsleep_front/assets/70925118/05c11306-ee42-4b64-b798-3db2ceab0d8e)
![DeepSleepDemo_Main2](https://github.com/alacori/deepsleep_front/assets/70925118/a5d7d551-4ca6-4fe1-a7b3-c2639b2b4453)


## 🛌 **주제선정 및 프로젝트 배경**
![](https://github.com/alacori/deepsleep_front/assets/70925118/10aa0fdd-9db9-4cb8-828b-91428bc167f0)
![](https://github.com/alacori/deepsleep_front/assets/70925118/3998532d-6042-4bc7-b60a-3ad166d71828)
![](https://github.com/alacori/deepsleep_front/assets/70925118/c7db6254-6701-4cdb-a687-8130e58aa9ec)


수면 무호흡증은 수면 중 호흡이 불규칙하거나 멈추는 장애로, 발생 빈도가 증가하고 위험성이 커지고 있습니다. 이로 인해 뇌졸중, 심근경색, 사망 등 다양한 위험이 증가하며 수면의 질 저하 및 주간 피로도 초래됩니다. 현재 진단은 병원에서의 복잡한 수면다원검사가 필요하지만, 이는 비용과 시간적 부담이 큽니다. 또한 1인 가구 증가로 수면 무호흡증 자각이 줄어드는 상황입니다. 따라서 스마트워치를 이용한 간편한 정보 수집으로 수면 무호흡증을 조기 예측하고 집에서도 관리할 수 있는 서비스를 개발하여, 치료와 삶의 질 개선에 기여하고자 하는 목적에서 "스마트워치를 이용한 수면 무호흡증 예측 서비스" 주제를 선정하게 되었습니다.
<br>
<br>

## 📈 **PSG 분석**

![](https://github.com/alacori/deepsleep_front/assets/70925118/f5fb1a60-981f-4186-a6bf-61664eb3e7f7)

수면다원검사(PSG) 분석에서는 스마트워치로 측정 가능한 변수들을 기반으로 AHI와 연관성이 있는 변수를 추출해 분석을 진행하고, 최적의 모델을 찾아 수면 무호흡증의 여부와 더불어 중증도를 쉽게 알 수 있는 웹 서비스를 제공합니다.
<br>
<br>

## 🫀 **ECG 분석**

![](https://i.ibb.co/pWFrbZr/our-goal.png)

ECG 분석에서는 심전도의 심장 박동과 관련된 다양한 정보를 추출합니다. 그 중에서도 심박변이율(HRV)은 심장 박동의 불규칙성을 나타내는 지표로 활용됩니다. 이렇게 추출된 HRV 정보를 기반으로 수면 무호흡증을 예측하는 모델을 구축합니다. 이 모델은 수면과 호흡의 상관관계를 분석하여, 수면 중 무호흡증 발생 가능성을 예측하는 데 도움을 줍니다. 따라서 ECG 분석은 수면 무호흡증 예측 분야에서 매우 중요한 역할을 수행합니다.
<br>
<br>

## 🎯 **기대효과**

![](https://i.ibb.co/pWFrbZr/our-goal.png)

먼저 스마트워치를 활용하여 간편하고 경제적인 방법으로 수면 무호흡증 예측이 가능합니다. 이는 복잡하고 비용이 많이 드는 수면다원검사(PSG) 대신 더 편리하게 수면 건강을 모니터링 할 수 있을 것입니다.<br>
또한, 스마트워치와 웹 서비스를 통한 AHI 지수 예측과 위험도 정보 제공으로 수면 무호흡증의 조기 발견이 가능할 것입니다. 조기에 문제를 파악하고 조치함으로써 심각한 합병증을 피하고 건강한 수면을 유지할 수 있습니다.<br>
마지막으로 스마트워치와 웹 서비스를 통한 AHI 모니터링으로 수면 무호흡증의 변화를 지속적으로 추적할 수 있습니다. 이를 통해 개인의 치료 효과를 평가하고 개선 여부를 확인할 수 있을 것입니다.
<br>
<br>

## 🛠 **Project Architecture**


![](https://github.com/alacori/deepsleep_front/assets/70925118/8b6efaf7-bd2c-469a-a051-e6e0d2746a1c)

<br>


## 📱**Feature Screen shots**

<br> <br>

|<img src="https://cdn.discordapp.com/attachments/1114839224361955328/1114839531846377522/4a9595dfd36dec45.gif">|<img src="https://cdn.discordapp.com/attachments/1114839224361955328/1114839531489865790/dc30a03f920bc47a.gif" >|<img src="https://cdn.discordapp.com/attachments/1114839224361955328/1114839349649997824/bee6b9e2f0221593.gif">|<img src="https://cdn.discordapp.com/attachments/1114839224361955328/1114839349238972476/9edf67f1937bfac5.gif">|
|------|------|------|------|
|**Medication & Meal Alarm**|**Medication Check**|**Connecting with the elderly**|**Interaction with the elderly**|


<br> <br>



## 👩‍💻 **Contributors**

### **Team 3**

|[정시은](https://github.com/ohyujeong)|[주윤나](https://github.com/ofzlo)|[한준호](https://github.com/alacori)|[박유진](https://github.com/dayoung20)|[신예진](https://github.com/ohyujeong)|[최성림](https://github.com/dayoung20)|
|---|---|---|---|---|---|
|<img src="https://github.com/alacori/deepsleep_front/assets/70925118/7ad78f84-e390-444b-9607-6d4e064796a8">|<img src="https://github.com/alacori/deepsleep_front/assets/70925118/d3d3306a-7895-4553-8af0-a55ec13b180a">|<img src="https://github.com/alacori/deepsleep_front/assets/70925118/c637a24c-1dbd-40d0-ad47-6f222f69f109">|<img src="https://cdn.discordapp.com/attachments/1091211029360422973/1091253631673708595/KakaoTalk_20230331_155008862.png">|
|PSG analysis|PSG analysis|PSG analysis|ECG analysis|ECG analysis|ECG analysis|
