#!/usr/bin/env Rscript
# Pairwise validity-rate differences heatmap (BH-adjusted p-values)
#
# Required (choose one):
#   --counts_csv PATH  CSV with generator, n_valid, n_total (aliases supported)
#   --agg_log PATH     aggregate_results.py stdout file
#
# Optional:
#   --quiz_batch_tag TAG    Parse the tagged summary block (e.g., Jan2026)
#   --out PATH              Output plot file (default: pairwise_p_heatmap.png)
#   --alpha FLOAT           Mark cells with '*' if BH-adjusted p < alpha (default: 0.01)
#   --order_by rate|name    Order axes by validity rate (default) or generator name
#   --palette NAME          Colorcet palette name (default: r2)
#   -h, --help              Show help
#
# Notes:
#   - This script EXCLUDES generator == "_EXAMPLE" by default.

usage <- function() {
  cat(paste(
    "Generate a heatmap of BH-adjusted pairwise proportion-test p-values for validity rates.",
    "",
    "Required (choose one):",
    "  --counts_csv PATH     CSV with columns: generator, n_valid, n_total",
    "  --agg_log PATH        Text file containing aggregate_results.py validity summary",
    "",
    "Optional:",
    "  --quiz_batch_tag TAG  Parse the tagged summary block (e.g., Jan2026)",
    "  --out PATH            Output plot file (default: pairwise_p_heatmap.png)",
    "  --alpha FLOAT         Mark cells with '*' if BH-adjusted p < alpha (default: 0.01)",
    "  --order_by rate|name  Order axes by validity rate (default) or generator name",
    "  --palette NAME        colorcet palette name (default: r2)",
    "  -h, --help            Show this help",
    "",
    "Examples:",
    "  Rscript r_analysis/validity_by_generator_heatmap.R --agg_log agg.txt --quiz_batch_tag Jan2026",
    "  Rscript r_analysis/validity_by_generator_heatmap.R --counts_csv r_analysis/validity_counts.csv --out heatmap.png",
    "",
    sep = "\n"
  ))
}

parse_args <- function(args) {
  opts <- list(
    out = "pairwise_p_heatmap.png",
    alpha = 0.01,
    order_by = "rate",
    palette = "r2"
  )
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (arg %in% c("-h", "--help")) {
      opts[["help"]] <- TRUE
      i <- i + 1
      next
    }

    if (!grepl("^--", arg)) {
      stop(paste("Unrecognized argument:", arg))
    }

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
    i <- i + 1
  }
  opts
}

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0(
      "Package '", pkg, "' is required.\n",
      "Install it with: install.packages('", pkg, "')"
    ))
  }
}

normalize_counts <- function(df) {
  if (!is.data.frame(df) || nrow(df) == 0) stop("No rows found in counts input.")

  names(df) <- tolower(names(df))

  if (!("generator" %in% names(df))) {
    if ("generator_model" %in% names(df)) df$generator <- df$generator_model
    else stop("Missing column 'generator' (or alias 'generator_model').")
  }

  if (!("n_valid" %in% names(df))) {
    if ("valid" %in% names(df)) df$n_valid <- df$valid
    else stop("Missing column 'n_valid' (or alias 'valid').")
  }

  if (!("n_total" %in% names(df))) {
    if ("total" %in% names(df)) df$n_total <- df$total
    else stop("Missing column 'n_total' (or alias 'total').")
  }

  df <- df[, c("generator", "n_valid", "n_total")]
  df$generator <- trimws(as.character(df$generator))
  df$n_valid <- suppressWarnings(as.integer(df$n_valid))
  df$n_total <- suppressWarnings(as.integer(df$n_total))

  if (any(df$generator == "" | is.na(df$generator))) stop("Blank/NA generator values found.")
  if (any(is.na(df$n_valid) | is.na(df$n_total))) stop("NA values found in n_valid/n_total.")
  if (any(df$n_total <= 0)) stop("All n_total values must be > 0.")
  if (any(df$n_valid < 0 | df$n_valid > df$n_total)) stop("All n_valid must satisfy 0 <= n_valid <= n_total.")

  # Aggregate if duplicate generators
  if (any(duplicated(df$generator))) {
    df <- aggregate(cbind(n_valid, n_total) ~ generator, data = df, FUN = sum)
    if (any(df$n_valid < 0 | df$n_valid > df$n_total)) {
      stop("After aggregation, found invalid rows where n_valid > n_total.")
    }
  }

  df$n_invalid <- df$n_total - df$n_valid
  df$validity_rate <- df$n_valid / df$n_total
  df$generator <- as.factor(df$generator)
  df
}

read_counts_csv <- function(path) {
  if (is.null(path) || path == "") stop("Missing --counts_csv PATH")
  if (!file.exists(path)) stop(paste("counts_csv not found:", path))
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  normalize_counts(df)
}

read_counts_from_agg_log <- function(path, quiz_batch_tag = NULL) {
  if (is.null(path) || path == "") stop("Missing --agg_log PATH")
  if (!file.exists(path)) stop(paste("agg_log not found:", path))

  lines <- readLines(path, warn = FALSE)

  label <- if (!is.null(quiz_batch_tag) && quiz_batch_tag != "") {
    paste0("Questions passing judge filters (tagged manifests: ", quiz_batch_tag, ")")
  } else {
    "Questions passing judge filters"
  }
  header <- paste0("[INFO] ", label)

  start_idx <- grep(header, lines, fixed = TRUE)
  if (length(start_idx) == 0) {
    stop(paste(
      "Could not find validity summary header in log:",
      header,
      "Tip: run quizbench/aggregate_results.py with judge filtering enabled and capture stdout.",
      sep = "\n"
    ))
  }

  i <- start_idx[[1]] + 1
  gens <- character()
  n_valid <- integer()
  n_total <- integer()

  pattern <- "^\\s*(.+?):\\s*valid=([0-9]+)\\s*/\\s*total=([0-9]+)\\s*\\("
  while (i <= length(lines)) {
    line <- lines[[i]]
    if (trimws(line) == "") break

    m <- regmatches(line, regexec(pattern, line))[[1]]
    if (length(m) == 4) {
      gen <- trimws(m[[2]])
      if (gen != "TOTAL") {
        gens <- c(gens, gen)
        n_valid <- c(n_valid, as.integer(m[[3]]))
        n_total <- c(n_total, as.integer(m[[4]]))
      }
    }
    i <- i + 1
  }

  df <- data.frame(generator = gens, n_valid = n_valid, n_total = n_total, stringsAsFactors = FALSE)
  normalize_counts(df)
}

# Robustly obtain a colorcet palette vector.
get_colorcet_palette <- function(palette_name, n = 256) {
  require_pkg("cetcolor")

  ns <- asNamespace("cetcolor")

  # Helper: try to generate palette with cet_pal using different argument names
  try_cet_pal <- function(name) {
    if (!exists("cet_pal", envir = ns, mode = "function")) return(NULL)
    f <- get("cet_pal", envir = ns)

    # Try a few common signatures; swallow errors and return NULL on failure
    attempts <- list(
      function() f(n, name),
      function() f(n, palette = name),
      function() f(n = n, name = name),
      function() f(n = n, palette = name)
    )
    for (a in attempts) {
      cols <- tryCatch(a(), error = function(e) NULL)
      if (!is.null(cols) && length(cols) > 0) return(cols)
    }
    NULL
  }

  # Helper: try to pull palette from a list object colorcet::colorcet
  try_colorcet_list <- function(name) {
    if (!exists("cetcolor", envir = ns)) return(NULL)
    pal_list <- get("cetcolor", envir = ns)
    if (!is.list(pal_list)) return(NULL)
    if (!(name %in% names(pal_list))) return(NULL)
    cols <- pal_list[[name]]
    if (length(cols) != n) cols <- grDevices::colorRampPalette(cols)(n)
    cols
  }

  # 1) Try exact name first (e.g., CET_R2_r)
  cols <- try_cet_pal(palette_name)
  if (is.null(cols)) cols <- try_colorcet_list(palette_name)
  if (!is.null(cols)) return(cols)

  # 2) If name ends with _r, try base palette and reverse
  if (grepl("_r$", palette_name)) {
    base <- sub("_r$", "", palette_name)

    cols2 <- try_cet_pal(base)
    if (is.null(cols2)) cols2 <- try_colorcet_list(base)
    if (!is.null(cols2)) return(rev(cols2))
  }

  stop(paste0(
    "Could not access cetcolor palette '", palette_name, "'.\n",
    "Try installing/updating cetcolor, or pass a different --palette name."
  ))
}

# ---------------- main ----------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || any(args %in% c("-h", "--help"))) {
  usage()
  quit(status = 0)
}

opts <- parse_args(args)
if (isTRUE(opts[["help"]])) {
  usage()
  quit(status = 0)
}

counts_csv <- opts[["counts_csv"]]
agg_log <- opts[["agg_log"]]
quiz_batch_tag <- opts[["quiz_batch_tag"]]
if ((is.null(quiz_batch_tag) || quiz_batch_tag == "") && !is.null(opts[["batch_tag"]])) {
  quiz_batch_tag <- opts[["batch_tag"]]
}

out <- opts[["out"]]
alpha <- suppressWarnings(as.numeric(opts[["alpha"]]))
if (is.na(alpha) || alpha <= 0 || alpha >= 1) stop("--alpha must be a number between 0 and 1 (exclusive).")

order_by <- tolower(as.character(opts[["order_by"]]))
if (!(order_by %in% c("rate", "name"))) stop("--order_by must be one of: rate, name")

palette_name <- as.character(opts[["palette"]])

if ((is.null(counts_csv) || counts_csv == "") && (is.null(agg_log) || agg_log == "")) {
  usage()
  stop("Provide exactly one of --counts_csv PATH or --agg_log PATH")
}
if (!is.null(counts_csv) && counts_csv != "" && !is.null(agg_log) && agg_log != "") {
  usage()
  stop("Provide only one of --counts_csv PATH or --agg_log PATH (not both).")
}

gen_counts <- if (!is.null(counts_csv) && counts_csv != "") {
  read_counts_csv(counts_csv)
} else {
  read_counts_from_agg_log(agg_log, quiz_batch_tag = quiz_batch_tag)
}

# EXCLUDE _EXAMPLE
gen_counts <- gen_counts[as.character(gen_counts$generator) != "_EXAMPLE", , drop = FALSE]

if (nrow(gen_counts) < 2) stop("After excluding '_EXAMPLE', need at least 2 generators to compute pairwise tests.")

# Name vectors for labeled output
x <- setNames(gen_counts$n_valid, as.character(gen_counts$generator))
n <- setNames(gen_counts$n_total, as.character(gen_counts$generator))

pw <- pairwise.prop.test(x, n, p.adjust.method = "BH")

# Build full symmetric BH-adjusted p-value matrix
gens <- names(x)
k <- length(gens)
P <- matrix(1, nrow = k, ncol = k, dimnames = list(gens, gens))
diag(P) <- NA_real_

for (r in rownames(pw$p.value)) {
  for (c in colnames(pw$p.value)) {
    p <- pw$p.value[r, c]
    if (!is.na(p)) {
      P[r, c] <- p
      P[c, r] <- p
    }
  }
}

# Order axes
if (order_by == "rate") {
  ord <- as.character(gen_counts$generator[order(gen_counts$validity_rate, decreasing = TRUE)])
} else {
  ord <- sort(as.character(gen_counts$generator))
}
P <- P[ord, ord, drop = FALSE]

# Print p-values to stdout (BH-adjusted)
cat("\nBH-adjusted pairwise p-values (unique pairs, sorted):\n")
idx <- which(upper.tri(P), arr.ind = TRUE)
pairs_df <- data.frame(
  gen1 = rownames(P)[idx[, 1]],
  gen2 = colnames(P)[idx[, 2]],
  p_adj = P[idx],
  stringsAsFactors = FALSE
)
pairs_df <- pairs_df[!is.na(pairs_df$p_adj), , drop = FALSE]
pairs_df <- pairs_df[order(pairs_df$p_adj), , drop = FALSE]
print(pairs_df, row.names = FALSE)

# Long form (keep diagonal with NA fill for a complete square)
df <- as.data.frame(as.table(P), stringsAsFactors = FALSE)
colnames(df) <- c("g_row", "g_col", "p_adj")

eps <- 1e-300
df$neglog10 <- ifelse(is.na(df$p_adj), NA_real_, -log10(pmax(df$p_adj, eps)))
df$sig <- ifelse(is.na(df$p_adj), FALSE, df$p_adj < alpha)
df$star <- ifelse(df$sig, "*", "")

# Keep only diagonal + lower triangle (avoid redundant symmetric cells)
df$row_idx <- match(df$g_row, ord)
df$col_idx <- match(df$g_col, ord)
df <- df[df$row_idx >= df$col_idx, , drop = FALSE]

# Factors so the plot looks like a matrix (top row = first in ord)
df$g_row <- factor(df$g_row, levels = rev(ord))
df$g_col <- factor(df$g_col, levels = ord)

# Palette
cols <- get_colorcet_palette(palette_name, n = 256)

# Plot
require_pkg("ggplot2")
library(ggplot2)

title_tag <- "Pairwise validity-rate differences"

# Legend ticks in p-value space (more interpretable than -log10 ticks)
format_p_break_label <- function(p) {
  if (!is.finite(p) || is.na(p)) return(NA_character_)
  if (p >= 1e-4) {
    return(sub("\\.?0+$", "", format(p, scientific = FALSE, trim = TRUE)))
  }
  s <- format(p, scientific = TRUE, digits = 1)
  s <- sub("e-0", "e-", s)
  s <- sub("e\\+0", "e+", s)
  s
}

# Clamp extremely small/large p-values to keep the legend readable.
# - Values below p_floor show as "<p_floor"
# - Values above p_cbar_cap show as ">=p_cbar_cap"
p_floor <- 1e-5
p_cbar_cap <- max(0.05, alpha)
min_fill <- -log10(p_cbar_cap)
max_fill <- -log10(p_floor)
df$neglog10_plot <- pmin(pmax(df$neglog10, min_fill), max_fill)

p_candidates <- c(0.1, 0.05, 0.01, 0.001, 0.0001)
p_breaks <- sort(unique(c(p_candidates, p_floor)), decreasing = TRUE)
p_breaks <- p_breaks[p_breaks <= p_cbar_cap & p_breaks >= p_floor]

fill_breaks <- -log10(p_breaks)
fill_labels <- vapply(p_breaks, format_p_break_label, character(1))
fill_labels[[1]] <- paste0(">=", fill_labels[[1]])
fill_labels[[length(fill_labels)]] <- paste0("<", fill_labels[[length(fill_labels)]])

p <- ggplot(df, aes(x = g_col, y = g_row, fill = neglog10_plot)) +
  geom_tile() +
  geom_text(aes(label = star), size = 5) +
  coord_equal() +
  scale_fill_gradientn(
    colours = cols,
    limits = c(min_fill, max_fill),
    breaks = fill_breaks,
    labels = fill_labels,
    guide = guide_colorbar(
      barheight = grid::unit(5, "cm"),
      title.position = "top",
      title.hjust = 0.5
    ),
    na.value = "grey95"
  ) +
  labs(
    title = title_tag,
    x = NULL,
    y = NULL,
    fill = "BH-adjusted p",
    # caption = paste0(
    #   "* indicates BH-adjusted p < ", alpha,
    #   ".\nFill scale clamps p to [", format_p_break_label(p_floor), ", ", format_p_break_label(p_cbar_cap), "]."
    # )
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(size = 18, hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, margin = margin(t = 4)),
    axis.text.y = element_text(size = 12, margin = margin(r = 4)),
    axis.ticks = element_line(color = "grey60", linewidth = 0.5),
    axis.ticks.length = grid::unit(3, "pt"),
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 12),
    legend.justification = "center",
    plot.margin = margin(t = 12, r = 14, b = 14, l = 14),
    panel.grid = element_blank()
  )

ggsave(filename = out, plot = p, width = 6.9, height = 6.0, units = "in", dpi = 300)

cat("\nPer-generator validity counts (excluding _EXAMPLE):\n")
print(gen_counts[, c("generator", "n_valid", "n_total", "validity_rate")], row.names = FALSE)
cat("\nSaved heatmap to:", out, "\n")
