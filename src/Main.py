from src.Model import train_model


def main():
    train_model('./data/dataset.xls', 10)
    #test_trained_model(10)

if __name__ == '__main__':
    main()