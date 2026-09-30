import splitfolders

splitfolders.ratio('dataset', output='split_dataset',
                    seed=42, ratio=(0.7, 0.15, 0.15))

print("Split complete.")