prediction = model.predict(img)
print(f'The result is probably:{ np.argmax(prediction)}')
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()