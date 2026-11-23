install.packages(c("pROC", "PRROC", "ggrepel"))
library(dplyr)
library(tidyr)
library(pROC)
library(PRROC)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(stringr)
library(mgcv)

################################################################
############## SOME FUNCTIONS ##################################
################################################################

safe_auc <- function(y, s) {
  y <- as.integer(y)
  if (length(unique(y)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(y, s, quiet = TRUE)))
}

safe_pr_auc <- function(y, s, return_curve = FALSE) {
  y <- as.integer(y)
  if (length(unique(y)) < 2) {
    if (return_curve) return(NULL)
    return(NA_real_)
  }
  pos <- s[y == 1]
  neg <- s[y == 0]
  pr <- PRROC::pr.curve(scores.class0 = pos, scores.class1 = neg, curve = return_curve)
  if (return_curve) {
    ## PRROC returns: recall, precision, threshold
    out <- as.data.frame(pr$curve)
    names(out) <- c("recall", "precision", "threshold")
    return(out)
  } else {
    return(unname(pr$auc.integral))
  }
}

roc_curve_df <- function(y, s) {
  y <- as.integer(y)
  if (length(unique(y)) < 2) return(NULL)
  roc_obj <- pROC::roc(y, s, quiet = TRUE)
  tibble(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities
  )
}

nef_at_chi <- function(y, s, chi = 0.05, decreasing = TRUE) {
  y <- as.integer(y)
  N <- length(y)
  if (N == 0) return(NA_real_)
  n_pos <- sum(y == 1)
  if (n_pos == 0) return(NA_real_) ## undefined if no actives
  
  ord <- if (decreasing) order(-s) else order(s)
  k <- max(1L, floor(chi * N)) ## ensures at least top-1
  hits_topk <- sum(y[ord][seq_len(k)] == 1)
  
  ef <- hits_topk / (chi * n_pos)
  ef_max <- min(1/chi, N / n_pos)
  if (!is.finite(ef_max) || ef_max <= 0) return(NA_real_)
  nef <- ef / ef_max
  as.numeric(pmin(pmax(nef, 0), 1))
}

nef_with_baseline <- function(y, s, chi = 0.05, decreasing = TRUE) {
  # Returns a one-row tibble with NEF, baseline NEF under random ranking, and dNEF
  # y: binary labels (0/1); s: scores
  
  y <- as.integer(y)
  N <- length(y)
  if (N == 0) {
    return(tibble(NEF = NA_real_, NEF_random = NA_real_, delta_NEF = NA_real_,
                  EF = NA_real_, EFmax = NA_real_, k = NA_integer_, hits_topk = NA_integer_))
  }
  n_pos <- sum(y == 1L)
  if (n_pos == 0L) {
    # Undefined when there are no actives
    return(tibble(NEF = NA_real_, NEF_random = NA_real_, delta_NEF = NA_real_,
                  EF = NA_real_, EFmax = NA_real_, k = NA_integer_, hits_topk = NA_integer_))
  }
  
  ord <- if (decreasing) order(-s) else order(s)
  k <- max(1L, floor(chi * N))  # ensure at least top-1
  hits_topk <- sum(y[ord][seq_len(k)] == 1L)
  
  EF <- hits_topk / (chi * n_pos)
  EFmax <- min(1/chi, N / n_pos)
  if (!is.finite(EFmax) || EFmax <= 0) {
    return(tibble(NEF = NA_real_, NEF_random = NA_real_, delta_NEF = NA_real_,
                  EF = EF, EFmax = EFmax, k = k, hits_topk = hits_topk))
  }
  NEF <- (EF / EFmax) |> pmin(1) |> pmax(0) |> as.numeric()
  NEF_random <- 1 / EFmax # random EF = 1; random NEF = 1/EFmax
  delta_NEF <- NEF - NEF_random
  
  tibble(NEF = NEF, NEF_random = NEF_random, delta_NEF = delta_NEF,
         EF = EF, EFmax = EFmax, k = k, hits_topk = hits_topk)
}

recall_at_chi <- function(y, s, chi = 0.10, decreasing = TRUE) {
  y <- as.integer(y)
  N <- length(y)
  if (N == 0) return(NA_real_)
  
  n_pos <- sum(y == 1L)
  if (n_pos == 0L) return(NA_real_)
  
  ord <- if (decreasing) order(-s) else order(s)
  k <- max(1L, floor(chi * N))
  
  ## How many positives are in the top k?
  hits_topk <- sum(y[ord][seq_len(k)] == 1L)
  
  hits_topk / n_pos
}

##########################################################################
################# HELPER CODE
##########################################################################

## Make pretty names
pretty_names <- c(
  "catnip/catnip" = "a-KG",
  "vergne_nitrilase/vergne_nitrilase" = "NTL"
)

standardize_fusionesp <- function(path, model_name = "FusionESP") {
  read.table(path, header = TRUE, sep = ",", quote = "", fill = NA, comment.char = "") %>%
    
    mutate(screening_set = str_replace_all(screening_set, pretty_names)) %>%
    
    transmute(
      screening_set = screening_set, 
      enzyme_id = as.character(SEQ),
      substrate_id = as.character(standardized_smiles), 
      label = as.integer(label), 
      score = as.numeric(y_pred), 
      model = model_name
      )
}

standardize_ezspec <- function(path, screening_set_name, model_name = "EZSpecificity") {
  read.table(path, header = TRUE, sep = ",", quote = "", fill = NA, comment.char = "") %>%
    
    transmute(
      screening_set = screening_set_name,
      enzyme_id = as.character(`Enzyme.Index`),
      substrate_id = as.character(`Substrate.Index`),
      label = as.integer(Label),
      score = as.numeric(score),
      model = model_name
    )
}

##########################################################################
################## READING IN ############################################
##########################################################################

## Note that this obvi hardcoded path won't work if run in this repo
## This file needs output from FusionESP and EZSpec to run so it has been added
## to both repos. 
fusionesp_path <- "~/mit_work/cyp_paper/1_NEW_screens/fusionesp_predictions/screening_predictions.csv"

ezspec_catnip_path <- "~/mit_work/cyp_paper/1_NEW_screens/ezspec_predictions/catnip_example_out.csv"
ezspec_nitrilase_path <- "~/mit_work/cyp_paper/1_NEW_screens/ezspec_predictions/vergne_nitrilase_example_out.csv"

df_fusionesp <- standardize_fusionesp(
  path = fusionesp_path,
  model_name = "FusionESP"
)

df_ezspec_catnip <- standardize_ezspec(
  path = ezspec_catnip_path,
  screening_set_name = "a-KG",
  model_name = "EZSpecificity"
)

df_ezspec_nitrilase <- standardize_ezspec(
  path = ezspec_nitrilase_path,
  screening_set_name = "NTL",
  model_name = "EZSpecificity"
)

df <- bind_rows(
  df_fusionesp,
  df_ezspec_catnip,
  df_ezspec_nitrilase
) %>%
  filter(!is.na(label), !is.na(score), !is.na(enzyme_id), !is.na(screening_set))

df <- df %>%
  mutate(
    model_short = case_when(
      model == "EZSpecificity" ~ "EZSp",
      model == "FusionESP" ~ "FESP",
      TRUE ~ model
    ),
    class_short = screening_set,
    model_class = paste(model_short, class_short)
  )

## Exclude any enzyme with no true positives
df <- df %>%
  group_by(model, screening_set, enzyme_id) %>%
  filter(sum(label == 1) > 0 & sum(label == 0) > 0) %>%
  ungroup()

################################################################
############## 1) METRICS, MACRO/MiCRO/RECALL@K ######################
################################################################

metrics_df <- df %>%
  group_by(model, screening_set, enzyme_id) %>%
  summarise(
    AUROC = safe_auc(label, score),
    AUPR = safe_pr_auc(label, score),
    .groups = "drop"
  )

class_summary <- metrics_df %>%
  group_by(model, screening_set) %>%
  summarise(
    mean_macro_AUROC = mean(AUROC, na.rm = TRUE),
    median_macro_AUROC = median(AUROC, na.rm = TRUE),
    mean_macro_AUPR = mean(AUPR, na.rm = TRUE),
    median_macro_AUPR = median(AUPR, na.rm = TRUE),
    n_macro = sum(!is.na(AUROC) & !is.na(AUPR)),
    .groups = "drop"
  )

micro_summary <- df %>%
  group_by(model, screening_set) %>%
  summarise(
    micro_AUROC = safe_auc(label, score),
    micro_AUPR = safe_pr_auc(label, score),
    n_enzymes_micro = n_distinct(enzyme_id),
    n_sub = n_distinct(substrate_id),
    n_pairs = n(),
    prevalence = mean(label == 1),
    recall_at_10 = recall_at_chi(label, score),
    recall_at_20 = recall_at_chi(label, score, chi = 0.2),
    .groups = "drop"
  )

class_summary <- class_summary %>%
  left_join(micro_summary, by = c("model", "screening_set"))

### Quantifying "good" macro AUPR
macro_aupr_quant <- df %>%
  group_by(model, screening_set, enzyme_id) %>%
  summarise(
    n_tested = n(),
    n_pos = sum(label == 1L),
    n_neg = sum(label == 0L),
    prevalence = mean(label == 1L),
    AUROC = safe_auc(label, score),
    AUPR = safe_pr_auc(label, score),
    .groups = "drop"
  ) %>%
  mutate(
    AUPR_over_random = AUPR / prevalence
  ) %>%
  group_by(model, screening_set) %>%
  summarise(
    pct_macro_AUPR_gt_random = mean(AUPR_over_random > 1, na.rm = T) * 100,
    pct_macro_AUPR_gt_2x_random = mean(AUPR_over_random > 2, na.rm = T) * 100,
    .groups = "drop"
  )

class_summary <- class_summary %>%
  left_join(macro_aupr_quant, by = c("model", "screening_set"))

write.table(
  class_summary,
  file = "~/mit_work/cyp_paper/1_NEW_screens/supplementary_table1_class_summary.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

################################################################
############## 2) POOLED MICRO ROC/PR CURVES ###################
################################################################

roc_curves <- df %>%
  group_by(model, screening_set) %>%
  reframe(roc_curve_df(label, score)) %>%
  drop_na()

pr_curves <- df %>%
  group_by(model, screening_set) %>%
  group_modify(~{
    crv <- safe_pr_auc(.x$label, .x$score, return_curve = TRUE)
    if (is.null(crv)) return(tibble())
    as_tibble(crv)
  }) %>%
  drop_na()

pr_baseline <- df %>%
  group_by(model, screening_set) %>%
  summarise(prevalence = mean(label == 1), .groups = "drop")

################################################################
############## 3) PLOT! ########################################
################################################################

### Adding labels of overall AUROC/AUPR to plot
roc_label_df <- micro_summary %>%
  mutate(
    curve_label = paste0(
      screening_set, " | ", model,
      " | AUROC=", sprintf("%.3f", micro_AUROC)
    )
  ) %>%
  select(model, screening_set, curve_label, micro_AUROC)

pr_label_df <- micro_summary %>%
  mutate(
    curve_label = paste0(
      screening_set, " | ", model,
      " | AUPR=", sprintf("%.3f", micro_AUPR)
    )
  ) %>%
  select(model, screening_set, curve_label, micro_AUPR)

roc_curves <- roc_curves %>%
  left_join(roc_label_df, by = c("model", "screening_set"))

pr_curves <- pr_curves %>%
  left_join(pr_label_df, by = c("model", "screening_set"))

pr_baseline_segments <- pr_baseline %>%
  left_join(pr_label_df, by = c("model", "screening_set")) %>%
  mutate(x0 = 0, x1 = 1, y0 = prevalence, y1 = prevalence)

################################################################
############## 4-PANEL BABY PLOTS ####################

palette <- c(
  "a-KH" = "#2A363BFF",
  "NTL"  = "#019875FF"
)

## AUROC labels
roc_labels <- roc_curves %>%
  distinct(screening_set, model, curve_label) %>%
  mutate(
    label = str_extract(curve_label, "AUROC=\\d+\\.\\d+"),
    label = str_replace(label, "AUROC=", ""),
    x = 0.08,
    y = case_when(
      screening_set == "a-KG" ~ 0.92,
      screening_set == "NTL"  ~ 0.86,
      TRUE ~ 0.80
    ),
    xend = x + 0.08,
    yend = y
  )

## AUPR labels
pr_labels <- pr_curves %>%
  distinct(screening_set, model, curve_label) %>%
  mutate(
    label = str_extract(curve_label, "AUPR=\\d+\\.\\d+"),
    label = str_replace(label, "AUPR=", ""),
    x = 0.08,
    y = case_when(
      screening_set == "a-KG" ~ 0.225,
      screening_set == "NTL"  ~ 0.205,
      TRUE ~ 0.185
    ),
    xend = x + 0.08,
    yend = y
  )

make_roc_plot <- function(model_name) {
  roc_df <- roc_curves %>%
    filter(model == model_name)
  
  lab_df <- roc_labels %>%
    filter(model == model_name)
  
  ggplot(roc_df, aes(x = fpr, y = tpr, color = screening_set, group = screening_set)) +
    geom_path(linewidth = 0.8, alpha = 0.9) +
    
    ## Random baseline
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    
    ## Small colored line samples for AUROC labels
    geom_segment(data = lab_df, aes(x = x, xend = xend, y = y, 
                                    yend = yend, color = screening_set),
                 linewidth = 0.9, inherit.aes = F) +
    
    ## AUROC text
    geom_text(data = lab_df, aes(x = xend + 0.015, y = y, 
                                 label = label, color = screening_set),
              hjust = 0, vjust = 0.5, size = 3.4, inherit.aes = F) +
    
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
    labs(
      title = paste0(model_name, " ROC"),
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Enzyme Class"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.margin = margin(10, 20, 10, 10),
      legend.position = "bottom"
    ) +
    scale_color_manual(values = palette)
}

make_pr_plot <- function(model_name) {
  pr_df <- pr_curves %>%
    filter(model == model_name)
  
  base_df <- pr_baseline_segments %>%
    filter(model == model_name)
  
  lab_df <- pr_labels %>%
    filter(model == model_name)
  
  ggplot(pr_df, aes(x = recall, y = precision, color = screening_set, group = screening_set)) +
    geom_path(linewidth = 0.8, alpha = 0.9) +
    
    ## Random / prevalence baselines
    geom_segment(data = base_df, aes(x = x0, xend = x1, y = y0, yend = y1, 
                                     color = screening_set, group = screening_set),
                 linetype = "dashed", linewidth = 0.6, inherit.aes = F) +
    
    ## Small colored line samples for AUPR labels
    geom_segment(data = lab_df, aes(x = x, xend = xend, y = y, 
                                    yend = yend, color = screening_set),
                 linewidth = 0.9, inherit.aes = F) +
    
    ## AUPR text
    geom_text(data = lab_df, aes(x = xend + 0.015, y = y, label = label, 
                                 color = screening_set),
              hjust = 0, vjust = 0.5, size = 3.4, inherit.aes = F) +
    
    coord_fixed(ratio = 4, xlim = c(0, 1), ylim = c(0, 0.25), clip = "on") +
    labs(
      title = paste0(model_name, " PR"),
      x = "Recall",
      y = "Precision",
      color = "Enzyme Class"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.margin = margin(10, 20, 10, 10),
      legend.position = "bottom"
    ) +
    scale_color_manual(values = palette)
}

p_roc_ezsp <- make_roc_plot("EZSpecificity")
p_roc_fesp <- make_roc_plot("FusionESP")

p_pr_ezsp <- make_pr_plot("EZSpecificity")
p_pr_fesp <- make_pr_plot("FusionESP")

micro_four_panel <- p_roc_ezsp | p_roc_fesp | p_pr_ezsp | p_pr_fesp
print(micro_four_panel)

## Per-enzyme (macro) AUROC and AUPR

p_box_auroc <- ggplot(
  metrics_df,
  aes(x = model, y = AUROC, fill = screening_set)
) +
  geom_boxplot(
    outlier.shape = NA,
    position = position_dodge(width = 0.75)
  ) +
  geom_jitter(
    position = position_jitterdodge(
      jitter.width = 0.12,
      dodge.width = 0.75
    ),
    alpha = 0.35,
    size = 1,
    color = "gray55"
  ) +
  coord_fixed(ylim = c(0, 1)) +
  labs(
    title = "Per-enzyme (macro) AUROC",
    x = "Model",
    y = "AUROC",
    fill = "Enzyme Class"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_manual(values = palette)

p_box_aupr <- ggplot(
  metrics_df,
  aes(x = model, y = AUPR, fill = screening_set)
) +
  geom_boxplot(
    outlier.shape = NA,
    position = position_dodge(width = 0.75)
  ) +
  geom_jitter(
    position = position_jitterdodge(
      jitter.width = 0.12,
      dodge.width = 0.75
    ),
    alpha = 0.35,
    size = 1,
    color = "gray55"
  ) +
  coord_fixed(ylim = c(0, 1)) +
  labs(
    title = "Per-enzyme (macro) AUPR",
    x = "Model",
    y = "AUPR",
    fill = "Enzyme Class"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_manual(values = palette)

## Layout: top row = pooled curves; bottom row = per-enzyme distributions
augmented_plot <- micro_four_panel / (p_box_auroc | p_box_aupr) 

print(augmented_plot)


#### NEF and recall@k
#### Computes dNEF10, dNEF20, recall@10, and recall@20
#### One row per enzyme / model / class / metric

early_rank_metrics <- df %>%
  group_by(model, screening_set, enzyme_id) %>%
  group_modify(~{
    out10 <- nef_with_baseline(.x$label, .x$score, chi = 0.10, decreasing = TRUE)
    out20 <- nef_with_baseline(.x$label, .x$score, chi = 0.20, decreasing = TRUE)
    
    tibble(
      n_tested = nrow(.x),
      n_pos = sum(.x$label == 1L),
      n_neg = sum(.x$label == 0L),
      prevalence = mean(.x$label == 1L),
      
      delta_NEF10 = out10$delta_NEF,
      delta_NEF20 = out20$delta_NEF,
      
      NEF10 = out10$NEF,
      NEF20 = out20$NEF,
      
      NEF10_random = out10$NEF_random,
      NEF20_random = out20$NEF_random,
      
      recall10 = recall_at_chi(.x$label, .x$score, chi = 0.10, decreasing = TRUE),
      recall20 = recall_at_chi(.x$label, .x$score, chi = 0.20, decreasing = TRUE),
      
      k10 = out10$k,
      k20 = out20$k,
      
      hits_top10 = out10$hits_topk,
      hits_top20 = out20$hits_topk
    )
  }) %>%
  ungroup() %>%
  mutate(
    model_short = case_when(
      model == "EZSpecificity" ~ "EZSp",
      model == "FusionESP" ~ "FESP",
      TRUE ~ model
    ),
    class_short = case_when(
      screening_set == "⍺-KH" ~ "a-KG",
      screening_set == "alpha-KH" ~ "a-KG",
      TRUE ~ screening_set
    ),
    model_class = paste(model_short, class_short)
  )

early_rank_long <- early_rank_metrics %>%
  select(model, screening_set, class_short, enzyme_id, n_tested, n_pos, n_neg,
         prevalence, delta_NEF10, delta_NEF20, recall10, recall20) %>%
  pivot_longer(
    cols = c(delta_NEF10, delta_NEF20, recall10, recall20),
    names_to = "metric", values_to = "value"
    ) %>%
  mutate(
    metric_label = case_when(
      metric == "delta_NEF10" ~ "Delta NEF@10%",
      metric == "delta_NEF20" ~ "Delta NEF@20%",
      metric == "recall10" ~ "Recall@10%",
      metric == "recall20" ~ "Recall@20%",
      TRUE ~ metric
    ),
    metric_label = factor( metric_label, levels = c(
      "Delta NEF@10%", "Delta NEF@20%", "Recall@10%", "Recall@20%"
      )
    )
  )

baseline_df <- tibble(
  metric_label = factor(
    c("Delta NEF@10%", "Delta NEF@20%", "Recall@10%", "Recall@20%"),
    levels = c("Delta NEF@10%", "Delta NEF@20%", "Recall@10%", "Recall@20%")
  ),
  baseline = c(0, 0, 0.10, 0.20)
)

early_rank_plot <- ggplot(
  early_rank_long %>% filter(!is.na(value)),
  aes(x = model, y = value, color = class_short, fill = class_short)) +
  geom_hline(data = baseline_df, aes(yintercept = baseline), linetype = "dashed",
             color = "black", inherit.aes = FALSE) +
  geom_boxplot(outlier.shape = NA, alpha = 0.45, width = 0.55, linewidth = 0.45,
               position = position_dodge(width = 0.75)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.12, jitter.height = 0.015,
                                             dodge.width = 0.75), alpha = 0.35, size = 1.4) +
  facet_wrap(~ metric_label, nrow = 1, scales = "free_y") +
  labs(
    title = "Early enrichment and recall per enzyme",
    x = "Model",
    y = "Metric value",
    color = "Enzyme class",
    fill = "Enzyme class"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  ) +
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette)

bottom_row <- (p_box_auroc | p_box_aupr | early_rank_plot) +
  plot_layout(widths = c(1, 1, 2.7), heights = c(1.2, 1.2, 1))

full_plot <- micro_four_panel / bottom_row +
  plot_layout(heights = c(1, 1.15))
pdf("~/mit_work/cyp_paper/1_NEW_screens/figures/figure1.pdf", width = 16, height = 8)
print(full_plot)
dev.off()

## Make supplememtary table for early ranking metrics

early_enrichment_quant <- early_rank_metrics %>%
  select(model, screening_set, enzyme_id, delta_NEF10, delta_NEF20, recall10, recall20) %>%
  pivot_longer(
    cols = c(delta_NEF10, delta_NEF20),
    names_to = "metric",
    values_to = "delta_NEF"
  ) %>%
  mutate(
    k = case_when(
      metric == "delta_NEF10" ~ "10%",
      metric == "delta_NEF20" ~ "20%",
      TRUE ~ NA_character_
    )
  ) %>%
  group_by(model, screening_set, k) %>%
  summarise(
    n_enzymes = sum(!is.na(delta_NEF)),
    n_above_random_dNEF = sum(delta_NEF > 0, na.rm = TRUE),
    pct_above_random_dNEF = mean(delta_NEF > 0, na.rm = TRUE) * 100,
    median_dNEF = median(delta_NEF, na.rm = TRUE),
    q25_dNEF = quantile(delta_NEF, 0.25, na.rm = TRUE),
    q75_dNEF = quantile(delta_NEF, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

print(early_enrichment_quant)

write.table(
  early_enrichment_quant,
  "~/mit_work/cyp_paper/1_NEW_screens/supptable2_ef_and_such.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

