library(tidyverse)
library(lubridate)
library(scales)
library(Cairo)
library (readxl)
library (reshape2)
library (dplyr)
library (ggplot2)
library (ggpubr)
library(pBrackets) 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

policies <- read_excel("policy-schedule.xlsx", range = "A1:O185")
policies$` ` = rep(NA,184)
policies.long <- melt(policies, variable.name = "Layer", value.name = "Relative transmission risk", id.vars = c("Date", "Policy name"))


# Calculate where to put the dotted lines that show up every three entries
#x.breaks <- seq(length(tasks$Task) + 0.5 - 3, 0, by=-3)

Policy = c("Initial response", "Lockdown", "Reopen 1", "Reopen 2", "Reopen 3", "Reopen 4", "Reopen 5")
Dates = c("2020-03-15 UTC", "2020-03-29 UTC", "2020-05-01 UTC", "2020-05-11 UTC", "2020-05-15 UTC", "2020-06-01 UTC", "2020-07-01 UTC")
tab <- data.frame(Dates, Policy)

fontfamily = 'Proxima Nova'

library(grid)
# Build plot
timeline <- ggplot(subset(policies.long, !(Layer %in% c("Overall beta", "Households"))), aes(x=Layer, y=date(Date), colour=`Relative transmission risk`)) + 
  geom_line(size=10) + 
  geom_hline(yintercept=date("2020-03-23 UTC"), colour="black", linetype="dotted") + 
  annotate("text", x=" ", y = date("2020-03-25 UTC"), label="Lockdown", hjust=0, family=fontfamily) +
  geom_hline(yintercept=date("2020-05-01 UTC"), colour="black", linetype="dotted") + 
  annotate("text", x=" ", y = date("2020-05-01 UTC"), label=" Phased reopening begins", hjust=0, family=fontfamily) +
  geom_hline(yintercept=date("2020-08-03 UTC"), colour="black", linetype="dotted") + 
  annotate("text", x=" ", y = date("2020-08-03 UTC"), label=" Victoria mandates\n masks", hjust=0, family=fontfamily) +
  theme_bw() + 
  annotate("text", x="Cafes and restaurants", y = date("2020-03-25 UTC"), label="Takeaway only", hjust=0, family=fontfamily) +
  annotate("text", x="Pubs and bars", y = date("2020-03-25 UTC"), label="Closed", hjust=0, family=fontfamily) +
  annotate("text", x="Large events", y = date("2020-03-25 UTC"), label="Cancelled", hjust=0, family=fontfamily) +
  annotate("text", x="Arts venues", y = date("2020-03-25 UTC"), label="Closed", hjust=0, family=fontfamily) +
  annotate("text", x="Socialising", y = date("2020-03-25 UTC"), label="Only 1 person visits", hjust=0, family=fontfamily) +
  annotate("text", x="Places of worship", y = date("2020-03-25 UTC"), label="Closed except for funerals", hjust=0, family=fontfamily) +
  annotate("text", x="Professional sport", y = date("2020-03-25 UTC"), label="Cancelled", hjust=0, family=fontfamily) +
  annotate("text", x="Community sport", y = date("2020-03-25 UTC"), label="Cancelled", hjust=0, family=fontfamily) +
  annotate("text", x="Public parks", y = date("2020-03-25 UTC"), label="Playgrounds/activities closed", hjust=0, family=fontfamily) +
  annotate("text", x="Workplaces", y = date("2020-03-25 UTC"), label="Work from home", hjust=0, family=fontfamily) +
  annotate("text", x="Schools", y = date("2020-03-25 UTC"), label="Home schooling", hjust=0, family=fontfamily) +
  labs(x=NULL, y=NULL) + 
  theme(axis.ticks.x = element_blank()) +
  coord_flip() +
  scale_colour_gradient(low = "white", high = "blue", na.value = NA, guide = guide_colorbar(barwidth=10)) +
  theme(legend.position="bottom",) +
  theme(text=element_text(size=14,  family=fontfamily, color="black")) + 
  scale_y_date(date_breaks="2 weeks", labels=date_format("%d %b")) #+
  #theme(axis.text.x=element_text(angle=45, hjust=1)) 
timeline

fn <- "policy-timelines.png"
ggsave(fn, plot = timeline, device="png", dpi=300)
