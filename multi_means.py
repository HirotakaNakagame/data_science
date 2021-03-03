def multi_means(s_feature, s_target, fillna="__MISSING__", pretty_print=True):
    """
    parameters
    ----------
        s_feature : pd.Series, Required
            feature series
        s_target : pd.Series, Reuqired
            target series. 0 = non responders & 1 = responders
        fillna : str type, default="__MISSING__"
            if there are missings, they will be filled with this
    return
    ------
        X_MM : pd.DataFrame
            DataFrame with count, percent, responder count, responder percent, index, and cum index
    """
    feature_name = s_feature.name
    target_name = s_target.name
    X_temp = pd.DataFrame({feature_name: s_feature, target_name: s_target})
    X_temp[feature_name].fillna(fillna, inplace=True)
    X_MM = pd.DataFrame(
        columns=["counts", "perc", "resp_counts", "resp_perc", "indices"],
        index=X_temp[feature_name].unique()
    )
    global_prob = X_temp[X_temp[target_name] == 1].shape[0] / X_temp.shape[0]

    __count = []
    __perc = []
    __resp_counts = []
    __resp_perc = []
    __index = []
    __cum_index = []

    for val in X_temp[feature_name].unique():
        __count.append(X_temp[X_temp[feature_name] == val].shape[0])
        __resp_counts.append(X_temp[(X_temp[feature_name] == val) & (X_temp[target_name] == 1)].shape[0])
        __resp_perc.append(round(
            X_temp[
                (X_temp[feature_name] == val) & (X_temp[target_name] == 1)
            ].shape[0] / X_temp[X_temp[target_name] == 1].shape[0] * 100
            , 2
        ))

    X_MM.counts = __count
    X_MM.perc = X_MM.counts.apply(lambda x: round(x / X_temp.shape[0] * 100, 2))
    X_MM.resp_counts = __resp_counts
    X_MM.resp_perc = __resp_perc
    X_MM.indices = round(X_MM.resp_perc / X_MM.perc * 100)

    if pretty_print is True:
        X_MM.counts = X_MM.counts.apply(prettyPrint)
        X_MM.resp_counts = X_MM.resp_counts.apply(prettyPrint)

    return X_MM
