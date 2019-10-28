# pylint:disable=W0312
import json, os, random
import numpy as np
from PIL import Image
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
# import sklearn.preprocessing as pre
from keras.models import Sequential
from keras.regularizers import l1

"""
x = np.array([
	[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
	[[3, 3, 4], [3, 4, 5], [4, 5, 6]],
	[[5, 4, 5], [4, 5, 6], [5, 6, 7]]
	])


def transpose(data):
	r_n = 0
	x = [[] for i in data[0]]
	for (c_i, c_v) in enumerate(x):
		for r in data:
			x[c_i].append(r[r_n])
		r_n += 1
	return x


# noinspection PyBroadException
def minmax_fit_3d(data):
	# print(data)
	data = data.tolist()
	mxmn = []
	for (r_i, row) in enumerate(data):
		mxmn.append([])
		roww = transpose(row)
		for (c_i, column) in enumerate(roww):
			mx, mn = max(column), min(column)
			mxmn[-1].append([mx, mn])
			for (i, v) in enumerate(column):
				try:
					nv = ((v - mn) / (mx - mn))
				except:
					# print(mx,mn,v)
					nv = 0
				column[i] = nv
			data[r_i] = transpose(roww)
	data = np.array(data)
	return {"data": data, "min_max": mxmn}


def minmax_reverse_3d(data, mxmn):
	data = data.tolist()
	for (r_i, row) in enumerate(data):
		roww = transpose(row)
		for (c_i, column) in enumerate(roww):
			mx, mn = mxmn[r_i][c_i][0], mxmn[r_i][c_i][1]
			for (i, v) in enumerate(column):
				nv = (mn + ((mx - mn) * v))
				column[i] = nv
			data[r_i] = transpose(roww)
	return np.array(data)


print(x)
fit = minmax_fit_3d(x)

print( fit["data"] )
print( fit["min_max"] )
print( minmax_reverse_3d(fit["data"], fit["min_max"] ) )
"""

"""
def min_maxify_3d(x):
	y = []
	mms = pre.MinMaxScaler()
	for (x_i, x_v) in enumerate(x.tolist()):
		#print(x_v)
		fitted = mms.fit_transform(x_v)
		y.append(fitted)
		#print(x_v," >> ",fitted)
	return np.array(y)

print(x)
print( min_maxify_3d(x) )
print( minmax_fit(x) )
"""
img_dir = "./imgs"
img_names = os.listdir(img_dir)
img_paths = ['{}/{}'.format(img_dir, name) for name in img_names]
img_data = {}
input_dim = 4
box_count = 0
# print(img_paths)

print("Fetching Data")
for (p_i, path) in enumerate(img_paths):
	# break
	f, e = os.path.splitext(path)
	outfile = f + ".jpg"

	if path != outfile:
		try:
			Image.open(path).save(outfile)
		except IOError:
			print("Error on '", path, "'. Image will be skipped over.")
			continue

	im = Image.open(path).convert("RGB")
	img_name = "{}/{}".format(im.size, outfile.split("/")[-1])
	# print(path, im.format, im.size, im.mode)

	if im.size[0] % input_dim > 0:
		im = im.resize((im.size[0] - (im.size[0] % input_dim), im.size[1]))
	if im.size[1] % input_dim > 0:
		im = im.resize((im.size[0], im.size[1] - (im.size[1] % input_dim)))

	# print(im.size)
	img_data[img_name] = {"scaled": [], "mm": []}
	for r in range(0, im.size[0], input_dim):
		for c in range(0, im.size[1], input_dim):
			box = (r, c, r + input_dim, c + input_dim)
			region = im.crop(box)
			np_region = np.array(region)
			data = (np_region / 255.0).flatten().tolist()
			# mm = minmax_fit_3d(np_region)
			# img_data[img_name]["mm"].append(mm["min_max"])
			img_data[img_name]["scaled"].append(data)  # mm["data"].tolist())
			box_count += 1
		if box_count >= 20000:
			box_count = 0
			break


	# im.show()
	# if p_i >= 0:
		# break


total_sample_len = [0, 0]
all_data = []
for (name, data) in img_data.items():
	print(name, ": >", len(data['scaled']), 'items')
	# print(data["scaled"][0]," > ",np.array(data["scaled"][0]).shape," > ",
	# len(data),"values\nmin max :",data["mm"][0])
	total_sample_len[0] += 1
	total_sample_len[1] += len(data["scaled"])
	# for (i, d) in enumerate(data["scaled"]):
	#	data["scaled"][i] = np.array(d).flatten()
	all_data += data["scaled"]

random.shuffle(all_data)
print("Gotten", total_sample_len[1], "samples from", total_sample_len[0],
      "images.")


x_train = np.array(all_data[:-300])
x_test = np.array(all_data[-300:-3])
x_sample = np.array(all_data[-3:-1])

print("train :", x_train.shape)
print("test :", x_test.shape)
print("sample :", x_sample.shape)

model = Sequential()
model.add(Dense(
	12,
	input_dim=48, use_bias=False,
	activation="linear", name="one"
	# activity_regularizer=l1(0.0001)
	))
model.add(Activation("relu"))
model.add(Dense(
	48, activation="linear",
	use_bias=False, name="two"
	))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

callbacks = [EarlyStopping(
	patience=30, verbose=1, min_delta=0.0000001,
	monitor='val_mean_absolute_error', mode="min"  # ,
	# restore_best_weights=True
	)]

fitted = model.fit(
	x_train, x_train,
	epochs=1000, batch_size=2048,
    validation_split=0.1, callbacks=callbacks)
print(fitted.history)

test = model.evaluate(x_test, x_test)
print("Test :", test)

model.save("encoder.h5")
print("Saved")

print(x_sample, " : ", x_sample.shape)
pred = np.array(model.predict(x_sample))
print(pred, " : ", pred.shape)
