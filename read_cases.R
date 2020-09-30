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

today = "2020/9/23"

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

locunknown <- data.frame(date = seq(as.Date("2020/6/1"), as.Date(today), "days")) %>%
  left_join(cases_by_source  %>% group_by(notification_date) %>% 
              filter(likely_source_of_infection %in% c("Locally acquired - source not identified"))%>%
              count(notification_date) %>% 
              rename(date=notification_date,new_diagnoses=n))  %>% replace(is.na(.), 0)
sum(locunknown$new_diagnoses)


#os <- data.frame(date = seq(as.Date("2020/3/9"), as.Date(today), "days")) %>%
#  left_join(cases_by_source  %>% group_by(notification_date) %>% 
#              filter(likely_source_of_infection %in% c("Interstate"))%>%
#              count(notification_date) %>% 
#              rename(date=notification_date,new_diagnoses=n))  %>% replace(is.na(.), 0)


nsw_epi_data <- read.csv("inputs/nsw_epi_data.csv")

ss$new_deaths <- c(nsw_epi_data$new_deaths[-(1:8)],rep(0,42))

write.csv(ss,"inputs/nsw_epi_data_os_removed.csv")  

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


pdata = data.frame(Scale=c(rep(0.2,8),rep(0.4,8),rep(0.6,8),rep(0.8,8),rep(1.0,8)))
pdata$Scale_label = factor(pdata$Scale, levels = c("0.2", "0.4", "0.6", "0.8", "1.0"),
                           labels = c("20% traceable","40% traceable","60% traceable","80% traceable","100% traceable"))
pdata$Scale_label[33:40] = rep("100% traceable",8)
  
pdata$Days = rep(seq(0,7),5)
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
    theme(legend.position = c(0.15, 0.813)) + 
    labs(fill = "") 
#    +    scale_fill_brewer(palette="Set2") 
)

fn <- "poisson_plot.pdf"
ggsave(fn, g, device="pdf", dpi=300)


