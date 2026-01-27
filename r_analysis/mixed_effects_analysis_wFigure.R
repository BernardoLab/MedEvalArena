#!/usr/bin/env Rscript
# Plot mean accuracies (across quizzes) by generator model with 95% CI.
#
# This script:
#   1) Computes accuracy per (model, quiz_id) = mean(correct) across questions
#   2) Computes, for each model, the mean of those quiz accuracies
#   3) Adds a 95% CI across quizzes (t-interval on quiz-level accuracies)
#   4) Plots a vertical bar chart sorted in descending order
#   5) Colors bars using the cetcolor package
#
# Usage:
#   Rscript dev/plot_model_accuracy_bar.R --input path/to/item_level.csv \
#     [--outdir R_results/accuracy_bar_YYYYMMDDTHHMMSS] \
#     [--palette r2] [--format png]
#
# Required columns in the input CSV (long format):
#   model, quiz_id, question_id, correct
#
# Notes on the CI:
#   The 95% CI is computed across quiz-level accuracies (each quiz weighted equally).
#   With 6 quizzes, this reflects between-quiz variability rather than item-level binomial SE.

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Package 'ggplot2' is required. Install with install.packages('ggplot2').")
}
if (!requireNamespace("cetcolor", quietly = TRUE)) {
  stop("Package 'cetcolor' is required. Install with install.packages('cetcolor').")
}
if (!requireNamespace("scales", quietly = TRUE)) {
  stop("Package 'scales' is required. Install with install.packages('scales').")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(cetcolor)
  library(scales)
})

usage <- function() {
  cat(paste(
    "Plot mean accuracies by generator model (across quizzes) with 95% CI.",
    "",
    "Required:",
    "  --input PATH             CSV with columns: model, quiz_id, question_id, correct",
    "",
    "Optional:",
    "  --outdir PATH            Output directory (default: R_results/accuracy_bar_<timestamp>)",
    "  --palette NAME           cetcolor palette name (default: r2)",
    "  --format EXT             Plot format: png or pdf (default: png)",
    "  -h, --help               Show this help",
    "",
    sep = "\n"
  ))
}

parse_args <- function(args) {
  opts <- list()
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      opts[["help"]] <- "true"
    } else if (grepl("^--", arg)) {
      if (grepl("=", arg)) {
        key <- sub("^--", "", sub("=.*", "", arg))
        val <- sub("^--[^=]+=", "", arg)
      } else {
        key <- sub("^--", "", arg)
        if (i == length(args)) stop(paste("Missing value for", arg))
        val <- args[[i + 1]]
        i <- i + 1
      }
      opts[[key]] <- val
    } else {
      stop(paste("Unrecognized argument:", arg))
    }
    i <- i + 1
  }
  opts
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || any(args %in% c("-h", "--help"))) {
  usage()
  quit(status = 0)
}

opts <- parse_args(args)
if (!is.null(opts[["help"]])) {
  usage()
  quit(status = 0)
}

input_path <- opts[["input"]]
if (is.null(input_path) || input_path == "") stop("Missing required --input PATH")
if (!file.exists(input_path)) stop(paste("Input file not found:", input_path))

timestamp <- format(Sys.time(), "%Y%m%dT%H%M%S")
outdir <- opts[["outdir"]]
if (is.null(outdir) || outdir == "") {
  outdir <- file.path("R_results", paste0("accuracy_bar_", timestamp))
}
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

palette_name <- opts[["palette"]]
if (is.null(palette_name) || palette_name == "") palette_name <- "r2"
palette_name <- cetcolor:::search_palettes(palette_name)

plot_format <- tolower(opts[["format"]])
if (is.null(plot_format) || plot_format == "") plot_format <- "png"
if (!(plot_format %in% c("png", "pdf"))) stop("--format must be 'png' or 'pdf'")

df <- read.csv(input_path, stringsAsFactors = FALSE)
required_cols <- c("model", "quiz_id", "question_id", "correct")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}
if (anyNA(df[, required_cols])) {
  stop("Missing values detected in required columns; please clean before plotting.")
}

# Normalize types
df$model <- as.factor(df$model)
df$quiz_id <- as.factor(df$quiz_id)
df$question_id <- as.factor(df$question_id)

# Normalize `correct` to 0/1 integer
if (is.logical(df$correct)) {
  df$correct <- as.integer(df$correct)
} else if (is.character(df$correct)) {
  vals <- tolower(trimws(df$correct))
  is_one <- vals %in% c("1", "true", "t", "yes", "y")
  is_zero <- vals %in% c("0", "false", "f", "no", "n")
  if (any(!(is_one | is_zero))) {
    stop("Column 'correct' must be binary (0/1 or true/false).")
  }
  df$correct <- ifelse(is_one, 1L, 0L)
}
if (!is.numeric(df$correct) || any(!(df$correct %in% c(0, 1)))) {
  stop("Column 'correct' must be binary (0/1).")
}

# 1) Quiz-level accuracies: mean(correct) within each (model, quiz_id)
quiz_acc <- aggregate(correct ~ model + quiz_id, data = df, FUN = mean)
names(quiz_acc)[names(quiz_acc) == "correct"] <- "quiz_accuracy"

# 2) Model-level mean & 95% CI across quizzes (t-interval)
ci_from_quiz_acc <- function(x, conf = 0.95) {
  n <- length(x)
  m <- mean(x)
  if (n < 2) {
    return(c(mean = m, lower = m, upper = m, n = n, sd = NA, se = NA))
  }
  s <- stats::sd(x)
  se <- s / sqrt(n)
  tcrit <- stats::qt((1 + conf) / 2, df = n - 1)
  half <- tcrit * se
  c(mean = m, lower = m - half, upper = m + half, n = n, sd = s, se = se)
}

spl <- split(quiz_acc$quiz_accuracy, quiz_acc$model)
model_stats_mat <- do.call(rbind, lapply(spl, ci_from_quiz_acc, conf = 0.95))
model_stats <- data.frame(
  model = rownames(model_stats_mat),
  mean_accuracy = as.numeric(model_stats_mat[, "mean"]),
  ci_lower = as.numeric(model_stats_mat[, "lower"]),
  ci_upper = as.numeric(model_stats_mat[, "upper"]),
  n_quizzes = as.integer(model_stats_mat[, "n"]),
  stringsAsFactors = FALSE
)

# Clamp CI to [0, 1] (accuracy bounds)
model_stats$ci_lower <- pmax(0, model_stats$ci_lower)
model_stats$ci_upper <- pmin(1, model_stats$ci_upper)

# 3) Sort descending for plotting
model_stats <- model_stats[order(-model_stats$mean_accuracy), ]
model_stats$model <- factor(model_stats$model, levels = model_stats$model)
model_stats$accuracy_label <- sprintf("%.1f", 100 * model_stats$mean_accuracy)  # e.g., "83.3"

# Save summaries
write.csv(quiz_acc, file.path(outdir, "quiz_accuracy_by_model.csv"), row.names = FALSE)
write.csv(model_stats, file.path(outdir, "model_mean_accuracy_across_quizzes.csv"), row.names = FALSE)

# 4) Colors from cetcolor (discrete palette sampled from a CET map)
n_models <- nrow(model_stats)
bar_cols <- cetcolor::cet_pal(n_models, name = palette_name)
names(bar_cols) <- levels(model_stats$model)

n_quizzes_total <- length(levels(df$quiz_id))

# 5) Plot
p <- ggplot(model_stats, aes(x = model, y = mean_accuracy, fill = model)) +
  geom_col(width = 0.82) +
  geom_text(
    aes(y = 0, label = accuracy_label),
    color = "white",
    size = 6,
    nudge_y = 0.3,  # 2 percentage points above the base
    vjust = 0
  ) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.22, linewidth = 0.6) +
  scale_fill_manual(values = bar_cols, guide = "none") +
  scale_y_continuous(
    limits = c(0, 1),
    labels = scales::label_percent(accuracy = 1),
    expand = expansion(mult = c(0, 0.06))
  ) +
  labs(
    title = "Mean accuracy by quiz-taking model",
    x = NULL,
    y = "Mean accuracy"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # plot.title.position = "plot",
    plot.title = element_text(hjust = 0.5),
    # gridlines:
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_line(linewidth = 0.6, colour = "grey70"),
    panel.grid.major.y = element_line(linewidth = 0.6, colour = "grey70")
  )

plot_file <- file.path(outdir, paste0("mean_accuracy_bar_with_95CI.", plot_format))
ggsave(plot_file, p, width = 7, height = 6, dpi = 320)

cat("Wrote:\n")
cat(" -", file.path(outdir, "quiz_accuracy_by_model.csv"), "\n")
cat(" -", file.path(outdir, "model_mean_accuracy_across_quizzes.csv"), "\n")
cat(" -", plot_file, "\n")
cat("Palette:", palette_name, "\n")
