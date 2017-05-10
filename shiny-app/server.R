# PYTHON_PATH = '/Users/Levittation/anaconda/bin/python'
PYTHON_PATH = '/usr/local/bin/python'

library(ggplot2)
library(plotly)
library(ggvis)
library(magrittr)


# Intialize global variables
assign("cnt", 1, envir = .GlobalEnv)
assign("df", NULL, envir = .GlobalEnv)
assign("cur_tick", "empty", envir = .GlobalEnv)


server <- function(input, output) {
  autoInvalidate <- reactiveTimer(500)  # Timer set to .5 seconds, for checking for updates
  scrollTimer <- reactiveTimer(500) # Timer for scroll speed TODO: Make this an input slider.
  observe({
    # Scroll backwards on click
    input$scrollBack 
    new_cnt <- cnt - 10
    assign("cnt", max(c(new_cnt, 1)), envir = .GlobalEnv)
  })
  
  observe({
    # Scroll backwards on click
    scrollTimer()
    if(input$stream) {
      assign("cnt", min(c((cnt+1), (nrow(df)-input$zoom))), envir = .GlobalEnv)
    }
  })
  
  observe({
    # Scroll forward on click
    input$scrollForward
    new_cnt <- cnt + 10
    assign("cnt", min(c(new_cnt, (nrow(df)-input$zoom))), envir = .GlobalEnv)
  })
  
  getNewData <- function(tick) {
    # Function to run python script when the ticker changes
    # Get ticker
    # Get file name
    tmp_data_file <- paste('/tmp/ticker_data_',tick, '.csv', sep='')
    tmp_trades_file <- paste('/tmp/tradelog_',tick, '.csv', sep='')
    # Call python script
    system(paste(PYTHON_PATH, "Updated.py", tick, sep=' '))
    # Wait for python script to finish running
    while(!file.exists(tmp_data_file)) {}
    # Read in python output
    tmp_data <- read.csv(tmp_data_file)
    tmp_trades <- read.csv(tmp_trades_file)
    if(is.na(tmp_data[1,1])) { # If output not good, set df to null
      assign("df", NULL, envir = .GlobalEnv)
    } else{  # Else assign python output to df
      names(tmp_data) = c('indx', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume')
      tmp_data <- tmp_data[rev(rownames(tmp_data)),]
      tmp_data$Date <- as.Date(tmp_data$Date, origin = "1970-01-01")
      assign("df", tmp_data, envir = .GlobalEnv)
    }
    if(is.na(tmp_trades[1,1])){
      assign('df_trades', NULL, envir= .GlobalEnv)
    } else {
      tmp_trades <- subset(tmp_trades, size !=0)
      if(nrow(tmp_trades) > 0) {
        tmp_trades$init_date <- as.Date(df$Date[tmp_trades$init_index], origin = "1970-01-01")
        tmp_trades$init_price <- df$Close[tmp_trades$init_index]
        tmp_trades$close_date <- as.Date(df$Date[tmp_trades$close_index], origin = "1970-01-01")
        tmp_trades$close_price <- df$Close[tmp_trades$close_index]
        assign('df_trades', tmp_trades, envir= .GlobalEnv)
      }
    }
    
  }
  
  # Render stock plot
  output$plot <- renderPlot({
    autoInvalidate()  # Look for new tick every .5 seconds
    new_tick <- input$ticker # Check if it's new, if it is, rebuild data
    if(cur_tick != new_tick) {
      getNewData(new_tick)
      assign("cur_tick", new_tick, envir = .GlobalEnv)
      assign("cnt", 1, envir = .GlobalEnv)
    }
    validate(
      need(!is.null(df), "Please Enter a Valid Ticker")  # Make sure the ticker is valid
    )
    # Plot
    plt <- ggplot(df[cnt:(cnt+input$zoom),], aes(x=Date, y=Close)) + 
      geom_line(size=1, aes(group=1)) +
      labs(title = paste("Trades on", new_tick), x='Date', y='Close') +
      ylim(floor(min(df$Close)), ceiling(max(df$Close))) +
      theme(
        plot.title = element_text(hjust=0.5, size=20),
        axis.title = element_text(size=15),
        axis.text.x = element_text(angle = 90, hjust = 1)
      )
    if(!is.null(df_trades)){
      bndry_date <- as.Date(df$Date[cnt+input$zoom])
      if(is.na(bndry_date)) bndry_date <- df$Date[nrow(df)]
      plt_trades <- df_trades %>% 
        subset(init_date < bndry_date) %>% 
        subset(close_date > df$Date[cnt])
      bfr_indx <- as.Date(plt_trades$close_date) > bndry_date
      plt_trades$close_date[bfr_indx] <- bndry_date
      plt_trades$close_price[bfr_indx] <- df$Close[cnt+input$zoom]
      plt_trades$init_date[plt_trades$init_date < df$Date[cnt]] <- df$Date[cnt]
      plt_trades$direction <- factor(ifelse(plt_trades$direction == 1, 'Buy', 'Short'))
      plt <- plt + 
        geom_segment(data=plt_trades, aes(color=factor(direction), 
                                          x = init_date,y = init_price, 
                                          xend=close_date,yend=close_price, 
                                          size=size)) +
        scale_size_continuous(range=c(0,3)) + 
        guides(size=guide_legend(title="Proportion of Wallet Bet"),
               color=guide_legend(title="Bet Type")) +
        scale_color_discrete()
      }
    plt
  })
  
  # Render info on brush
  output$info <- renderText({
    validate(
      need(!is.null(df_trades), "")
    )
    e <- input$plot_brush
    if(is.null(e)) return("Click and Drag to See More Information\n")
    
    d1 <- as.Date(e$xmin, origin = "1970-01-01")
    d2 <- as.Date(e$xmax, origin = "1970-01-01")
    x_msg <- paste("\n\tStart:", d1, "\n\tEnd:", d2)

    
    cur_trades <- df_trades %>% 
      subset(close_date < d2) %>% 
      subset(close_date > d1)
    
    selected_df <- df %>% 
      subset(Date < d2) %>% 
      subset(Date > d1)
    
    y_msg <- paste("\n\tMinimum: ", min(selected_df$Close), "\n\tMaximum:", max(selected_df$Close))
    
    sub_gain <- round(sum((cur_trades$close_price - cur_trades$init_price) * as.numeric(cur_trades$direction)), 3)
    
    paste0(
      "Close Price Range: ", 
      y_msg, 
      "\nDate Range: ", 
      x_msg,
      "\n Net Gain of Trades Closed Within Date Range: $", sub_gain
    )
  })
  
  output$summary <- renderText({
    scrollTimer()
    validate(
      need(!is.null(df_trades), "The Algorithm Did Not Make Any Trades on This Stock")
      )
    bndry_date <- as.Date(df$Date[cnt+input$zoom])
    if(is.na(bndry_date)) bndry_date <- df$Date[nrow(df)]
    cur_trades <- df_trades %>% 
      subset(close_date < bndry_date)
    total_gain <- sum((cur_trades$close_price - cur_trades$init_price) * as.numeric(cur_trades$direction))
    paste("Trades Net Gain To Date: $", round(total_gain, 3), sep='')
  })
  
  output$rawData <- renderDataTable({
    validate(
      need(!is.null(df), "Please Enter a Valid Ticker")  # Make sure the ticker is valid
    )
    df[,2:ncol(df)]
  })
  
  output$tradeData <- renderDataTable({
    validate(
      need(!is.null(df_trades), "The Algorithm Did Not Make Any Trades on This Stock")
    )
    names(df_trades) <- c('Bet_Type','Bet_Size', 'drp1', 'drp2', 'Start_Date', 'Start_Price',
                          'End_Date', 'End_Price')
    df_trades$drp1 <- NULL
    df_trades$drp2 <- NULL
    df_trades$profit <- (df_trades$End_Price - df_trades$Start_Price) * as.numeric(df_trades$Bet_Type)
    df_trades$Bet_Type <- factor(ifelse(df_trades$Bet_Type == 1, 'Buy', 'Short'))
    
    df_trades
  })
}