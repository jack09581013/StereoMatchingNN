import matplotlib.pyplot as plt
import pickle
import jacky_tool

history = jacky_tool.load('model/FlowNetS.ht')
print('Number of epochs:', len(history['loss']))

start = 0
plt.figure()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('EPE')
plt.plot(history['loss'][start:], label='Training')
plt.plot(history['val_loss'][start:], label='Testing')
plt.legend()
plt.show()