

weight = 0.5
input = 0.5
goal_prediction = 0.8 #이것빼곤 다 튜닝 된다 -> 이건 실직적 목표 
lr= 0.1
epochs = 100

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

