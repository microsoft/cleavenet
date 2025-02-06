import argparse
import datetime
import os

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm



import cleavenet
from cleavenet import plotter
from cleavenet.utils import get_data_dir, mmps

bhatia_mmps = ['MMP1', 'MMP10', 'MMP12', 'MMP13', 'MMP17', 'MMP3', 'MMP7']
bhatia_index = []
for i,m in enumerate(mmps):
    if m in bhatia_mmps:
        bhatia_index.append(i)

#parse from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", default=50, type=int,
                    help="number of epochs")
parser.add_argument("--model-type", default='transformer', type=str,
                    help="transformer or lstm architecture")
parser.add_argument("--max-len", default=10, type=int,
                    help="maximum sequence length to add to dataset")
parser.add_argument("--batch-size", default=128, type=int,
                    help="batch size range")
parser.add_argument("--learning-rate", default=0.005, type=float,
                    help="learning rate")
parser.add_argument("--d-model", default=32, type=int,
                    help="dimensions of model")
parser.add_argument("--log-freq", default=50, type=int,
                    help="frequency to log to tensorboard")
parser.add_argument("--save-freq", default=100, type=int,
                    help="frequency to save weights")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="smoothing rate for the exp filter")
parser.add_argument("--split", default=0.8, type=float,
                    help="train val split ratio for ensembling")
parser.add_argument("--regu", default=0.01, type=float,
                    help=" Regularization parameter for LSTM")
parser.add_argument("--ensemble", default=5, type=int,
                    help="Model iterations to use for ensemble uncertainty calculation")
args = parser.parse_args()

###################################################################################################
# Get data, split into train and test
###################################################################################################

# Get path to data
data_dir = get_data_dir()
data_path = os.path.join(data_dir, "kukreja.csv")

random_seed = list(range(args.ensemble))

###################################################################################################
# Function to run
###################################################################################################
def main():
    # Load in kukreja data
    kukreja = cleavenet.data.DataLoader(data_path, seed=0, task='regression', model=args.model_type, test_split=0.2,
                                        dataset='kukreja')

    bhatia = cleavenet.data.DataLoader(data_path, seed=0, task='regression', model=args.model_type, test_split=0,
                                       dataset='bhatia',
                                       use_dataloader=kukreja)
    x_bhatia = cleavenet.data.tokenize_sequences(bhatia.X, kukreja)
    if args.model_type == 'transformer':
        cls_idx = kukreja.char2idx[kukreja.CLS]
        x_bhatia = np.stack([np.append(np.array(cls_idx), s) for s in x_bhatia])
    y_bhatia = bhatia.y

    # Run ensemble training
    for ensemble in range(args.ensemble):
        # Train/valid splits for each ensemble, use pre-split data to preserve test set
        X_train, X_valid, y_train, y_valid = train_test_split(kukreja.X_train, kukreja.y_train, test_size=1-args.split,
                                                              random_state=random_seed[ensemble])
        vocab_size = len(kukreja.char2idx)
        print("vocab size", vocab_size)
        num_samples = len(X_train)
        num_valid_samples = len(X_valid)
        print("Training samples:", num_samples, "Validation samples: ", num_valid_samples)

        run_name = "run-%d" % ensemble
        print('--- Starting trial: %s' % run_name)

        # Build the predictor model
        if args.model_type == 'lstm':
            transformer=False
            embedding_dim= 22 # args.d_model
            dropout=0.25
            model = cleavenet.models.RNNPredictor(vocab_size, embedding_dim, args.d_model,
                                                  dropout, args.regu, args.max_len, len(mmps), mask_zero=True)
            lr = args.learning_rate # 0.005

        elif args.model_type == 'transformer':
            transformer=True
            num_layers=4
            num_heads=8
            dropout=0.01
            embedding_dim = 128
            model = cleavenet.models.TransformerEncoder(
                num_layers=num_layers,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=args.d_model,  # dense params
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=len(mmps),
                pool_outputs=True,
                mask_zero=True)
            lr = cleavenet.models.TransformerSchedule(args.d_model)

        print("Learning rate", lr)

        model_label='/'+args.model_type+'_'+str(ensemble)

        model.build((args.batch_size, None))
        model.summary()

        optimizer = tf.optimizers.Adam(lr)

        @tf.function  # comment out for eager execution (if you want to debug)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)  # forward pass
                loss = model.compute_loss(y, y_hat)  # compute loss
            grads = tape.gradient(loss, model.trainable_variables)  # compute gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update
            return loss, y_hat

        def smooth(prev, val):
            if prev is not None:
                new = (1 - args.alpha) * val + args.alpha * prev
            else:
                new = val
            return new

        global_step = 0
        running_loss = None
        running_rmse = None
        best_val_loss = float('inf')
        best_b_val_loss = float('inf')

        # LOGGING
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join('save' + model_label, '{}_PREDICTOR'.format(current_time))
        os.makedirs(save_dir)
        train_log_dir = os.path.join('logs' + model_label, '{}_PREDICTOR_train'.format(current_time))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs' + model_label, '{}_PREDICTOR_val'.format(current_time))
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(args.num_epochs):
            print("Epoch ", epoch)
            pbar = tqdm(range(num_samples // args.batch_size))
            for iter in pbar:
                # Grab a batch and train
                x, y = cleavenet.data.get_batch(X_train, y_train, args.batch_size, kukreja, transformer=transformer)
                loss, y_hat = train_step(x, y)
                rmse = model.compute_rmse(y, y_hat)  # compute train rmse

                running_loss = smooth(running_loss, loss.numpy())
                running_rmse = smooth(running_rmse, rmse.numpy())

                global_step += 1

                # saving
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=global_step)
                    tf.summary.scalar('rmse', rmse, step=global_step)

            if epoch > 0:  # run validation every epoch
                print("Running validation")
                vbar = tqdm(range(len(X_valid) // args.batch_size))
                val_loss = []
                val_rmse = []
                for v_iter in vbar:
                    xv, yv = cleavenet.data.get_batch(X_valid, y_valid, args.batch_size, kukreja, transformer=transformer)
                    yv_hat = model(xv, training=False)
                    val_loss.append(model.compute_loss(yv, yv_hat)*args.batch_size)  # compute loss
                    val_rmse.append(model.compute_rmse(yv, yv_hat)*args.batch_size) # compute val rmse
                val_loss = np.sum(val_loss)/len(X_valid) # batch-averaged loss
                val_rmse = np.sum(val_rmse)/len(X_valid)
                # saving
                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                    tf.summary.scalar('rmse', val_rmse, step=epoch)

                    # save weights only if validation loss decreases
                    print("best val loss:", best_val_loss)
                    if val_loss < best_val_loss:
                        print(f"Saving with val loss: {val_loss:.4f}")
                        print(f"Val rmse: {val_rmse:.4f}")
                        model.save_weights(os.path.join(save_dir, \
                                                        "{}.weights.h5".format("model")))
                        best_val_loss = val_loss

                ## run bhatia validation
                b_yv_hat = model(x_bhatia, training=False)
                b_yv_hat_condensed = tf.concat([tf.expand_dims(b_yv_hat[:,index], axis=1) for index in bhatia_index], axis=1)
                b_val_loss = model.compute_loss(y_bhatia[:, :len(bhatia_index)], b_yv_hat_condensed) # compute loss
                b_val_rmse = model.compute_rmse(y_bhatia[:, :len(bhatia_index)], b_yv_hat_condensed)
                print("bhatia loss:", b_val_loss)
                # saving
                with val_summary_writer.as_default():
                    tf.summary.scalar('b-loss', b_val_loss, step=epoch)
                    tf.summary.scalar('b-rmse', b_val_rmse, step=epoch)

                    # save weights only if validation loss decreases
                    print("best val loss:", best_b_val_loss)
                    if b_val_loss < best_b_val_loss:
                        print(f"Saving with bhatia val loss: {b_val_loss:.4f}")
                        print(f"bhatia Val rmse: {b_val_rmse}")
                        model.save_weights(os.path.join(save_dir, \
                                                        "{}.weights.h5".format("best-bhatia-model")))
                        best_b_val_loss = b_val_loss

        save_file = save_dir + '/best_loss.csv'
        with open(save_file, 'w') as f:
            f.write(str(best_val_loss))


        ##################################
        # After training assess performance of trained model in full set of test data
        # using load model here so we can use the best checkpoint
        ensemble_dir = save_dir
        checkpoint_path_final = os.path.join(ensemble_dir, "model.weights.h5")

        # Re-build the predictor model
        if args.model_type == 'lstm':
            model = cleavenet.models.RNNPredictor(vocab_size, embedding_dim, args.d_model,
                                                  dropout, args.regu, args.max_len, len(mmps))
        elif args.model_type == 'transformer':
            model = cleavenet.models.TransformerEncoder(
                num_layers=num_layers,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=args.d_model,  # dense params
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=len(mmps),
                pool_outputs=True)

        model.build((len(kukreja.X_test), None))
        model.summary()
        model.load_weights(checkpoint_path_final)  # load weights from best checkpoint


        xt, yt = cleavenet.data.get_batch(kukreja.X_test, kukreja.y_test, len(kukreja.X_test), kukreja, test=True, transformer=transformer)
        yt_hat = model(xt, training=False)  # forward pass
        embeddings = model.last_layer_embeddings
        test_rmse = model.compute_rmse(yt, yt_hat, axis=0)  # compute val rmse
        print(test_rmse)

        # Save embeddings for later
        np.save(os.path.join(ensemble_dir, 'test_weighted_cluster_embeddings.npy'), np.array(embeddings))

        # Plot results
        # Scatterplot of predicted vs true
        plotter.plot_parity(yt, yt_hat, mmps, ensemble_dir)
        # plot RMSE of all MMP families
        plotter.plot_rmse(test_rmse, mmps, ensemble_dir)

if __name__ == "__main__":
    main()
