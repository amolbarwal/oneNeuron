# oneNeuron
oneNeuron | Perceptron



## add image -
![sample Image](plots/and.png)

## add code -
```python
def main(data, eta, epochs, filename, plotfilename):

    df = pd.DataFrame(data)
    logging.info(f'This is actual dataframe \n{df}')

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotfilename, model)
```
``` bash
git add . && git commit -m "message" && git push origin main
```