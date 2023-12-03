from surprise import Dataset, Reader
from surprise.dump import load
from surprise import accuracy

reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))
test_data = Dataset.load_from_file('data/test.csv', reader=reader)
test = test_data.build_full_trainset().build_testset()

_, model = load('../models/svd_best.pickle')

train_data = Dataset.load_from_file('../data/interim/train.csv', reader=reader)
train = train_data.build_full_trainset()

model.fit(train)

predictions = model.test(test)

rmse = accuracy.rmse(predictions, verbose=False)
mse = accuracy.mse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)
fcp = accuracy.fcp(predictions, verbose=False)

print(f'RMSE: {rmse}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'FCP: {fcp}')
