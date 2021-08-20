

weight = 0.5
input = 0.5
goal_prediction = 0.8 #이것빼곤 다 튜닝 된다 -> 이건 실직적 목표 
lr= 0.1
epochs = 10

for iteration in range(epochs):
    prediction = input * weight
    error = (prediction - goal_prediction) **2
    print('weight : ', weight)
    print("Eroor : ", str(error) + "\tPrediction :" + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2
    print('up',up_error)
    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2
    print('down',down_error)

    if(down_error < up_error) :
        weight = weight -lr
    if(down_error > up_error) :
        weight = weight +lr


# weight :  0.5
# Eroor :  0.30250000000000005    Prediction :0.25
# up 0.25
# down 0.3600000000000001
# weight :  0.6
# Eroor :  0.25   Prediction :0.3
# up 0.20250000000000007
# down 0.30250000000000005
# weight :  0.7
# Eroor :  0.20250000000000007    Prediction :0.35
# up 0.16000000000000006
# down 0.25
# weight :  0.7999999999999999
# Eroor :  0.16000000000000006    Prediction :0.39999999999999997
# up 0.12250000000000007
# down 0.20250000000000007
# weight :  0.8999999999999999
# Eroor :  0.12250000000000007    Prediction :0.44999999999999996
# up 0.09000000000000007
# down 0.16000000000000006
# weight :  0.9999999999999999
# Eroor :  0.09000000000000007    Prediction :0.49999999999999994
# up 0.06250000000000006
# down 0.12250000000000007
# weight :  1.0999999999999999
# Eroor :  0.06250000000000006    Prediction :0.5499999999999999 
# up 0.04000000000000003
# down 0.09000000000000007
# weight :  1.2
# Eroor :  0.04000000000000003    Prediction :0.6
# up 0.022500000000000006
# down 0.06250000000000006
# weight :  1.3
# Eroor :  0.022500000000000006   Prediction :0.65
# up 0.009999999999999995
# down 0.04000000000000003
# weight :  1.4000000000000001
# Eroor :  0.009999999999999995   Prediction :0.7000000000000001
# up 0.0024999999999999935
# down 0.022500000000000006

