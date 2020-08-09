def predict_from_model(ROI_list,model,labels):
  for charac in ROI_list:
    charac = cv2.resize(charac[1],(80,80))
    charac = np.stack((charac,)*3, axis=-1)

    prediction = labels.inverse_transform([np.argmax(model.predict(charac[np.newaxis,:]))])
    title = np.array2string(prediction)
    print(title.strip("'[]"),end="")

    
    pred = model.predict(charac[np.newaxis,:])
    
    max_prob_index = heapq.nlargest(4,range(len(pred[0])),key = pred[0].__getitem__)
    
    max_prob_char = labels.inverse_transform(max_prob_index)

    max_prob_dict = {max_prob_char[i]:pred[0][max_prob_index[i]] for i in range(len(max_prob_char))}

    max_prob_list.append(max_prob_dict)