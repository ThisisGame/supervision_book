import roboflow

roboflow.login()

rf = roboflow.Roboflow()
project = rf.workspace('1310945803-qq-com').project('trafficsign-tkm0x')
dataset = project.version('4').download("yolov8")