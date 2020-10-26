import pandas as pd
import plotnine as pn

import analysis
import util

verbs = ["AllOpen", "BeCertain", "BelPart", "WondowLess"]
trials = range(30)

long_data = pd.DataFrame()
final_acc = pd.DataFrame()
for verb in verbs:
    verb_data = util.read_trials_from_csv(
        f"../single/{verb.lower()}/data", trials=trials
    )
    for trial in trials:
        accuracies = verb_data[trial][f"{verb}_accuracy"].values
        long_data = long_data.append(
            pd.DataFrame(
                {
                    "verb": verb,
                    "trial": trial,
                    "accuracy": analysis.smooth_data(accuracies, smooth_weight=0.5),
                    "step": verb_data[trial]["global_step"],
                }
            ),
            ignore_index=True,
        )
        final_acc = final_acc.append(
            {"verb": verb, "trial": trial, "accuracy": sum(accuracies[-5:]) / 5},
            ignore_index=True,
        )

print(long_data)

plot = pn.ggplot(long_data) + pn.geom_line(
    pn.aes(x="step", y="accuracy", color="verb", group="verb*trial"), alpha=0.5
)

print(plot)

plot = (
    pn.ggplot(final_acc, pn.aes(x="verb", y="accuracy"))
    + pn.geom_violin(pn.aes(fill="verb"))
    # + pn.geom_dotplot(dotsize=0.05, stackdir="centerwhole", binaxis="y", binwidth=0.0005)
    + pn.geom_point(size=0.5, alpha=0.5)
)
print(plot)