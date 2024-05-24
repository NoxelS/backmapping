from library.analysis.plots import plot_hist_single

# plot_names = [
#     "",
#     "training_loss.pdf",
#     "training_mse.pdf",
#     "training_lr.pdf",
#     "training_mae.pdf",
#     "training_val_acc.pdf",
#     "training_val_loss.pdf",
#     "training_val_mae.pdf",
# ]

# for i in range(1, 7):
#     plot_cluster_hist(i).savefig(os.path.join(os.path.join(config(Keys.DATA_PATH), "hist"), plot_names[i]), **{"dpi": 300, "bbox_inches": "tight"})
#     print(f"Saved {plot_names[i]}")

plot_hist_single(94)
