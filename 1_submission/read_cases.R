########################################
# Script for reading in NSW case data
#
# Date last modified: Aug 19, 2020
# For questions on this script, contact robyn@math.ku.dk
########################################

########################################
# Load libraries
########################################
library (ggplot2)
library (dplyr)
library (lubridate)
library (ggpubr)

today = "2020/9/30"

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
cases_by_source <- read.csv("confirmed_cases_table3_likely_source.csv")
cases_by_source$notification_date <- ymd(cases_by_source$notification_date)

ss <- data.frame(date = seq(as.Date("2020/3/9"), as.Date(today), "days")) %>%
  left_join(cases_by_source  %>% group_by(notification_date) %>% 
              filter(likely_source_of_infection %in% c("Locally acquired - contact of a confirmed case and/or in a known cluster", "Locally acquired - source not identified"))%>%
              count(notification_date) %>% 
              rename(date=notification_date,new_diagnoses=n))  %>% replace(is.na(.), 0)


all <- data.frame(date = seq(as.Date("2020/6/1"), as.Date(today), "days")) %>%
  left_join(cases_by_source  %>% group_by(notification_date) %>% 
              filter(likely_source_of_infection %in% c("Locally acquired - contact of a confirmed case and/or in a known cluster", "Locally acquired - source not identified", "Interstate","Overseas"))%>%
              count(notification_date) %>% 
              rename(date=notification_date,new_diagnoses=n))  %>% replace(is.na(.), 0)

sum(all$new_diagnoses)

# Get cases by source since June
sum_cases_by_source <- cases_by_source %>% group_by(likely_source_of_infection) %>% 
  filter(notification_date %in% seq(as.Date("2020/6/1"), as.Date(today), "days")) %>%
  count()



#os <- data.frame(date = seq(as.Date("2020/3/9"), as.Date(today), "days")) %>%
#  left_join(cases_by_source  %>% group_by(notification_date) %>% 
#              filter(likely_source_of_infection %in% c("Interstate"))%>%
#              count(notification_date) %>% 
#              rename(date=notification_date,new_diagnoses=n))  %>% replace(is.na(.), 0)


nsw_epi_data <- read.csv("nsw_epi_data.csv")

ss$new_deaths <- c(nsw_epi_data$new_deaths[-(1:8)],rep(0,49))

write.csv(ss,"nsw_epi_data_os_removed.csv")  

########################################
# Read testing data
########################################

today = "2020/9/16"

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rt <- read.csv("~/Documents/git/covasim-australia/applications/nsw/pcr_testing_table2_age_group.csv")

tt <- rt %>% group_by(test_date) %>% 
  count(test_date) %>% 
  rename(date=test_date,new_tests=n)  %>% replace(is.na(.), 0)


ss$new_tests <- c(tt$new_tests,rep(NA,3))
write.csv(ss,"inputs/nsw_epi_data_os_removed.csv")  

########################################
# Make poisson plot
########################################

library (ggplot2)

pdata = data.frame(Scale=c(rep(0.25,8),rep(0.5,8),rep(0.75,8),rep(1.0,8)))
pdata$Scale_label = factor(pdata$Scale, levels = c("0.25", "0.5", "0.75", "1.0"),
                           labels = c("25% traceable","50% traceable","75% traceable","100% traceable"))
pdata$Scale_label[25:32] = rep("100% traceable",8)
  
pdata$Days = rep(seq(0,7),4)
pdata$Tracing = ppois(pdata$Days,lambda = 1)*pdata$Scale
fontfamily="Optima"

(g <- ggplot(data=pdata, aes(x=Days, y=Tracing, colour=Scale_label)) +
    geom_line() + 
    #facet_wrap(~Scale_label) + 
    theme_bw() + 
    theme(text=element_text(size=16,  family=fontfamily)) + 
    theme(legend.background = element_rect(fill = "white"),
          legend.key = element_rect(fill = "white", color = NA))  +
    scale_y_continuous(labels = scales::percent) + 
    ylab("Percent traced") +
    xlab("Days after case notification") +
    theme(legend.title = element_blank()) + 
    theme(legend.position = c(0.11, 0.88)) + 
    labs(fill = "") 
#    +    scale_fill_brewer(palette="Set2") 
)

fn <- "poisson_plot.png"
ggsave(fn, g, device="png", dpi=300)


