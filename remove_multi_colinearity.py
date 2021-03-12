def remove_multicollinearity(X, target, threshold=0.5, target_threshold=0.05):
    print(f"#Initial inputs: {X.shape[1] - 1}")
    finals = []
    X_corr = X.corr().abs().sort_values(by=TARGET, ascending=False)
#     pbar = tqdm(X_corr.index)
    for i, col in enumerate(X_corr.index):
#         print(f"\t{i}: {col}")
        if col == TARGET:
            pass
        elif X_corr.loc[TARGET, col] < target_threshold:
            pass
        else:
            if len(finals) == 0:
                finals.append(col)
            else:
                clean = True
                while clean:
                    for var in finals:
                        if X_corr.loc[var, col] > threshold:
                            clean=False
                        else:
                            pass
                    break
                if clean:
                    finals.append(col)
    print(f"#Final inputs: {len(finals)}")
    print("Top Vars")
    print(X_corr.loc[finals, [TARGET]].head())
    print("Bottom Vars")
    print(X_corr.loc[finals, [TARGET]].tail())
    return finals
