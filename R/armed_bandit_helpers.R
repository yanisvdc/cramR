utils::globalVariables(c("inv", "Agent"))


# Ref: adapted from the contextual package

# SIMULATOR with reward beta parameters retrieval ----------------------------------------------------

Simulator <- R6::R6Class(
  "Simulator",
  class = FALSE,
  public = list(
    agents = NULL,
    workers = NULL,
    agent_count = NULL,
    horizon = NULL,
    simulations = NULL,
    worker_max = NULL,
    internal_history = NULL,
    save_context = NULL,
    save_theta = NULL,
    do_parallel = NULL,
    sims_and_agents_list = NULL,
    t_over_sims = NULL,
    set_seed = NULL,
    progress_file = NULL,
    log_interval = NULL,
    save_interval = NULL,
    include_packages = NULL,
    outfile = NULL,
    global_seed = NULL,
    chunk_multiplier = NULL,
    policy_time_loop = NULL,
    cl = NULL,
    initialize = function(agents,
                          horizon = 100L,
                          simulations = 100L,
                          save_context = FALSE,
                          save_theta = FALSE,
                          do_parallel = TRUE,
                          worker_max = NULL,
                          set_seed = 0,
                          save_interval = 1,
                          progress_file = FALSE,
                          log_interval = 1000,
                          include_packages = NULL,
                          t_over_sims = FALSE,
                          chunk_multiplier = 1,
                          policy_time_loop = FALSE) {

      # save current seed
      self$global_seed <- contextual::get_global_seed()

      if (!is.list(agents)) agents <- list(agents)

      self$progress_file <- progress_file
      self$log_interval <- as.integer(log_interval)
      self$horizon <- as.integer(horizon)
      self$simulations <- as.integer(simulations)
      self$save_theta <- save_theta
      self$save_context <- save_context
      self$agents <- agents
      self$agent_count <- length(agents)
      self$worker_max <- worker_max
      self$do_parallel <- do_parallel
      self$t_over_sims <- t_over_sims
      self$set_seed <- set_seed
      self$save_interval <- as.integer(save_interval)
      self$include_packages <- include_packages
      self$chunk_multiplier <- as.integer(chunk_multiplier)
      self$policy_time_loop <- policy_time_loop

      self$reset()
    },
    reset = function() {
      set.seed(self$set_seed)
      self$workers <- 1

      # create or clear log files
      if (self$progress_file) {
        cat(paste0(""), file = "workers_progress.log", append = FALSE)
        cat(paste0(""), file = "agents_progress.log", append = FALSE)
        cat(paste0(""), file = "parallel.log", append = FALSE)
        self$outfile <- "parallel.log"
      }

      # (re)create history data and meta data tables
      self$internal_history <- History$new()
      self$internal_history$set_meta_data("horizon",self$horizon)
      self$internal_history$set_meta_data("agents",self$agent_count)
      self$internal_history$set_meta_data("simulations",self$simulations)
      self$internal_history$set_meta_data("sim_start_time",format(Sys.time(), "%a %b %d %X %Y"))

      # unique policy name creation
      agent_name_list <- list()
      for (agent_index in 1L:self$agent_count) {
        current_agent_name <- self$agents[[agent_index]]$name
        agent_name_list <- c(agent_name_list,current_agent_name)
        current_agent_name_occurrences <-
          length(agent_name_list[agent_name_list == current_agent_name])
        if (current_agent_name_occurrences > 1) {
          self$agents[[agent_index]]$name <-
            paste0(current_agent_name,'.',current_agent_name_occurrences)
        }
        agent_name <-  self$agents[[agent_index]]$name
        bandit_name <- self$agents[[agent_index]]$bandit$class_name
        policy_name <- self$agents[[agent_index]]$policy$class_name
        self$internal_history$set_meta_data("bandit", bandit_name , group = "sim", agent_name = agent_name)
        self$internal_history$set_meta_data("policy", policy_name , group = "sim", agent_name = agent_name)
      }
    },
    run = function() {
      # set parallel or serial processing
      `%fun%` <- foreach::`%do%`

      # nocov start
      if (self$do_parallel) {
        self$register_parallel_backend()
        `%fun%` <- foreach::`%dopar%`

        # If Microsoft R, set MKL threads to 1

        # Due to an unresolved incompatibility between MRAN and RStudio:
        # https://github.com/rstudio/rstudio/issues/5933
        # https://social.technet.microsoft.com/Forums/en-US/2791e896-c284-4330-88f2-2dcd4acea074
        # setting MKL threads to 1 is disabled when running from RStudio.

        isRStudio <- Sys.getenv("RSTUDIO") == "1"
        if (!isRStudio && "RevoUtilsMath" %in% rownames(installed.packages())) {
          RevoUtilsMath::setMKLthreads(1)
        }
      }
      # nocov end

      # create a list of all sims (sims*agents), to be divided into chunks
      index <- 1
      sims_and_agents_list <- vector("list", self$simulations*self$agent_count)
      for (sim_index in 1L:self$simulations) {
        for (agent_index in 1L:self$agent_count) {
          sims_and_agents_list[[index]] <-
            list(agent_index = agent_index, sim_index   = sim_index)
          index <- index + 1
        }
      }

      # copy variables used in parallel processing to local environment
      horizon                  <- self$horizon
      agent_count              <- self$agent_count
      save_context             <- self$save_context
      save_theta               <- self$save_theta
      progress_file            <- self$progress_file
      save_interval            <- self$save_interval
      log_interval             <- self$log_interval
      t_over_sims              <- self$t_over_sims
      set_seed                 <- self$set_seed
      agents                   <- self$agents
      include_packages         <- self$include_packages
      policy_time_loop          <- self$policy_time_loop

      # calculate chunk size
      if (length(sims_and_agents_list) <= self$workers) {
        chunk_divider <- length(sims_and_agents_list)
      } else {
        chunk_divider <- self$workers * self$chunk_multiplier
      }
      # split sims vector into chuncks
      sa_iterator <- itertools::isplitVector(sims_and_agents_list, chunks = chunk_divider)
      # include packages that are used in parallel processes
      par_packages <- c(c("data.table","iterators","itertools"),include_packages)

      # some info messages
      message(paste("Simulation horizon:",horizon))
      message(paste("Number of simulations:",length(sims_and_agents_list)))
      message(paste("Number of batches:",chunk_divider))
      message("Starting main loop.")

      # start running the main simulation loop
      private$start_time <- Sys.time()
      foreach_results <- foreach::foreach(
        sims_agent_list = sa_iterator,
        i = iterators::icount(),
        .inorder = TRUE,
        .export = c("History","Formula"),
        .noexport = c("sims_and_agents_list","internal_history","sa_iterator"),
        .packages = par_packages
      ) %fun% {
        index <- 1L
        sim_agent_counter <- 0
        sim_agent_total <- length(sims_agent_list)

        # TODO: Can be done smarter and cleaner?
        multiplier <- 1
        for (sim_agent_index in sims_agent_list) {
          sim_agent <- agents[[sim_agent_index$agent_index]]
          if(isTRUE(sim_agent$bandit$arm_multiply))
            if(multiplier < sim_agent$bandit$k)
              multiplier <- sim_agent$bandit$k
        }
        allocate_space <- floor((horizon * sim_agent_total * multiplier) / save_interval) + sim_agent_total

        local_history <- History$new( allocate_space,
                                      save_context,
                                      save_theta)

        for (sim_agent_index in sims_agent_list) {
          sim_agent <- agents[[sim_agent_index$agent_index]]$clone(deep = TRUE)

          sim_agent$sim_index <- sim_agent_index$sim_index
          sim_agent$agent_index <- sim_agent_index$agent_index

          ###############################################################################################

          # Set sim_id explicitly for the bandit
          sim_agent$bandit$sim_id <- sim_agent_index$sim_index

          ###############################################################################################


          sim_agent_counter <- sim_agent_counter + 1
          if (isTRUE(progress_file)) {
            sim_agent$progress_file <- TRUE
            sim_agent$log_interval <- log_interval
            cat(paste0("[",format(Sys.time(), format = "%H:%M:%OS6"),"] ",
                       "        0 > init - ",sprintf("%-20s", sim_agent$name),
                       " worker ", i,
                       " at sim ", sim_agent_counter,
                       " of ", sim_agent_total,"\n"),
                file = "workers_progress.log", append = TRUE)
          }
          simulation_index <- sim_agent$sim_index
          agent_name <- sim_agent$name
          local_curent_seed <- simulation_index + set_seed * 42
          set.seed(local_curent_seed)
          sim_agent$bandit$post_initialization()
          sim_agent$policy$post_initialization()
          if(isTRUE(sim_agent$bandit$arm_multiply)) {
            if(policy_time_loop)
              horizon_loop <- horizon
            else
              horizon_loop <- horizon * sim_agent$bandit$k
            data_length <- horizon * sim_agent$bandit$k
          } else {
            horizon_loop <- horizon
            data_length <- horizon
          }
          set.seed(local_curent_seed + 1e+06)
          sim_agent$bandit$generate_bandit_data(n = data_length)

          if (isTRUE(t_over_sims)) sim_agent$set_t(as.integer((simulation_index - 1L) * horizon_loop))
          step <- list()

          loop_time <- 0L
          while (loop_time < horizon_loop) {
            step <- sim_agent$do_step()
            if(isTRUE(policy_time_loop)) {
              loop_time <- step[[5]]
            } else {
              loop_time <- loop_time + 1L
            }
            if (!is.null(step[[3]]) && ((step[[5]] == 1) || (step[[5]] %% save_interval == 0))) {
              local_history$insert(
                index,                                         #index
                step[[5]],                                     #policy_t
                step[[1]][["k"]],                              #k
                step[[1]][["d"]],                              #d
                step[[2]],                                     #action
                step[[3]],                                     #reward
                agent_name,                                    #agentname
                simulation_index,                              #sim
                if (save_context) step[[1]][["X"]] else NA,    #context
                if (save_theta) step[[4]] else NA              #theta
              )
              index <- index + 1L
            }
          }
        }
        sim_agent$bandit$final()
        local_history$data[t!=0]
      }

      # bind all results
      foreach_results <- data.table::rbindlist(foreach_results)
      foreach_results[, agent := factor(agent)]
      self$internal_history$set_data_table(foreach_results[sim > 0 & t > 0], auto_stats = FALSE)
      rm(foreach_results)
      private$end_time <- Sys.time()
      gc()
      message("Finished main loop.")

      self$internal_history$set_meta_data("sim_end_time",format(Sys.time(), "%a %b %d %X %Y"))
      formatted_duration <- contextual::formatted_difftime(private$end_time - private$start_time)
      self$internal_history$set_meta_data("sim_total_duration", formatted_duration)
      message(paste0("Completed simulation in ",formatted_duration))

      start_time_stats <- Sys.time()
      message("Computing statistics.")
      # update statistics TODO: not always necessary, add option arg to class?
      self$internal_history$update_statistics()

      # load global seed
      .Random.seed <- self$global_seed

      # set meta data and messages
      self$stop_parallel_backend()
      self$internal_history
    },
    register_parallel_backend = function() {
      # nocov start
      # setup parallel backend
      message("Setting up parallel backend.")
      nr_cores <- parallel::detectCores()
      if (nr_cores >= 3) self$workers <- nr_cores - 1
      if (!is.null(self$worker_max)) {
        if (self$workers > self$worker_max) self$workers <- self$worker_max
      }

      # make sure no leftover processes
      doParallel::stopImplicitCluster()


      if(!is.null(self$outfile)) {
        self$cl <- parallel::makeCluster(self$workers, useXDR = FALSE, type = "PSOCK",
                                         methods = FALSE, setup_timeout = 30, outfile = self$outfile)
      } else {
        self$cl <- parallel::makeCluster(self$workers, useXDR = FALSE, type = "PSOCK",
                                         methods = FALSE, setup_timeout = 30)
      }

      message(paste0("Cores available: ",nr_cores))
      message(paste0("Workers assigned: ",self$workers))
      doParallel::registerDoParallel(self$cl)
      # nocov end
    },
    stop_parallel_backend = function() {
      # nocov start
      if (self$do_parallel) {
        try({
          parallel::stopCluster(self$cl)
        })
        doParallel::stopImplicitCluster()
      }
      # nocov end
    }
  ),
  private = list(
    start_time = NULL,
    end_time = NULL,
    finalize = function() {
      # set global seed back to value before
      contextual::set_global_seed(self$global_seed)
      #closeAllConnections()
    }
  ),
  active = list(
    history = function(value) {
      if (missing(value)) {
        self$internal_history
      } else {
        warning("## history$data is read only", call. = FALSE)
      }
    }
  )
)


# Ref: contextual package: ---------------------------------------------------------------------

################################################################################################

################################################################################################

# BANDIT #######################################################################################

################################################################################################


#' @importFrom R6 R6Class
Bandit <- R6::R6Class(
  class    = FALSE,
  public   = list(
    k           = NULL,  # Number of arms (integer, required)
    d           = NULL,  # Dimension of context feature vector (integer, required)
    unique      = NULL,  # Vector of arm indices of unique context features (vector, optional)
    shared      = NULL,  # Vector of arm indices of context features shared between arms (vector, optional)
    class_name  = "Bandit",
    initialize  = function() {
      # Is called before the Policy instance has been cloned.
      # Initialize Bandit. Set self$d and self$k here.
    },
    post_initialization = function() {
      # Is called after a Simulator has cloned the Bandit instance [number_of_simulations] times.
      # Do sim level random generation here.
      invisible(self)
    },
    get_context = function(t) {
      stop("Bandit subclass needs to implement bandit$get_context()", call. = FALSE)
      # Return a list with number of arms self$k, number of feature dimensions self$d and, where
      # applicable, a self$d dimensional context vector or self$d x self$k dimensional context matrix X.
      list(X = context, k = arms, d = features) # nocov
    },
    get_reward = function(t, context, action) {
      stop("Bandit subclass needs to implement bandit$get_reward()", call. = FALSE)
      # Return a list with the reward of the chosen arm and, if available, optimal arm reward and index
      list(reward = reward_for_choice_made, optimal_reward = optimal_reward, optimal_arm = optimal_arm) # nocov
    },
    generate_bandit_data = function(n) {
      # Optionally pregenerate n contexts and rewards here.
    },
    final = function() {
      # called on object destruction
    }
  )
)


#############################################################################################

# AGENT

#############################################################################################

Agent <- R6::R6Class(
  "Agent",
  portable = FALSE,
  class = FALSE,
  public = list(
    policy = NULL,
    bandit = NULL,
    sim_index = NULL,
    agent_index = NULL,
    name = NULL,
    agent_t = NULL,
    policy_t = NULL,
    cum_regret = NULL,
    cum_reward = NULL,
    progress_file = NULL,
    log_interval = NULL,
    sparse = NULL,
    initialize = function(policy, bandit, name=NULL, sparse = 0.0) {
      self$bandit                 <- bandit
      self$policy                 <- policy
      self$sparse                 <- sparse
      if (is.null(name)) {
        self$name  <- gsub("Policy", "", policy$class_name)
      } else {
        self$name  <- name
      }
      self$reset()
      invisible(self)
    },
    reset = function() {
      if(is.null(self$bandit$d)) self$bandit$d = 1
      if(is.null(self$bandit$unique)) {
        self$bandit$unique <- c(1:self$bandit$d)
      }
      if(is.null(self$bandit$shared)) {
        self$bandit$shared <- c(1:self$bandit$d)
      }
      context_initial_params        <- list ()
      context_initial_params$d      <- self$bandit$d
      context_initial_params$k      <- self$bandit$k
      context_initial_params$unique <- self$bandit$unique
      context_initial_params$shared <- self$bandit$shared
      self$policy$set_parameters(context_initial_params)
      self$policy$initialize_theta(context_initial_params$k)
      self$progress_file <- FALSE
      self$log_interval <- 1000L
      cum_reward <<- 0.0
      cum_regret <<- 0.0
      agent_t <<- 0L
      policy_t <<- 1L
      invisible(self)
    },
    do_step = function() {

      agent_t  <<- agent_t + 1L
      context   <- bandit$get_context(agent_t)
      if(is.null(context)) return(list(context = NULL, action = NULL, reward = NULL))
      if(is.null(context$d)) context$d <- self$bandit$d
      if(is.null(context$unique)) context$unique <- c(1:context$d)
      if(is.null(context$shared)) context$shared <- c(1:context$d)
      action    <- policy$get_action(policy_t, context)
      reward    <- bandit$get_reward(agent_t, context, action)

      if (is.null(reward)) {
        theta   <- NULL
      } else {
        if (!is.null(reward[["optimal_reward"]])) {
          reward[["regret"]]      <- reward[["optimal_reward"]] - reward[["reward"]]
          cum_regret              <<- cum_regret + reward[["regret"]]
          reward[["cum_regret"]]  <- cum_regret
        } else {
          reward[["regret"]]      <- 0.0
          reward[["cum_regret"]]  <- 0.0
        }

        if (!is.null(reward[["context"]])) {
          context <- reward[["context"]]
        }

        cum_reward                <<- cum_reward + reward[["reward"]]
        reward[["cum_reward"]]    <- cum_reward

        if (self$sparse == 0.0 || runif(1) > self$sparse) {
          theta   <- policy$set_reward(policy_t, context, action, reward)
        } else {
          theta   <- policy$theta
        }
        if (!is.null(policy$is_oracle) && isTRUE(policy$is_oracle)) {
          reward$reward    <- theta$optimal_reward
          action$choice    <- theta$optimal_arm
        }
        policy_t  <<- policy_t + 1L
      }
      if(isTRUE(self$progress_file)) {
        if (agent_t %% self$log_interval == 0) {
          cat(paste0("[",format(Sys.time(), format = "%H:%M:%OS6"),"] ",sprintf("%9s", agent_t)," > step - ",
                     sprintf("%-20s", self$name)," running ",bandit$class_name,
                     " and ",policy$class_name,"\n"),file = "agents_progress.log", append = TRUE)
        }
      }
      list(context = context, action = action, reward = reward, theta = theta, policy_t = (policy_t-1))
    },
    set_t = function(t) {
      agent_t <<- t
      invisible(self)
    },
    get_t = function(t) {
      agent_t
    }
  )
)


#############################################################################################

# HISTORY

#############################################################################################


#' @importFrom data.table data.table as.data.table set setorder setkeyv copy uniqueN setcolorder tstrsplit
#' @import rjson
History <- R6::R6Class(
  "History",
  portable = FALSE,
  public = list(
    n            = NULL,
    save_theta   = NULL,
    save_context = NULL,
    context_columns_initialized = NULL,
    initialize = function(n = 1, save_context = FALSE, save_theta = FALSE) {
      self$n                           <- n
      self$save_context                <- save_context
      self$save_theta                  <- save_theta
      self$reset()
    },
    reset = function() {
      gc()
      self$context_columns_initialized <- FALSE
      self$clear_data_table()
      private$initialize_data_tables()
      invisible(self)
    },
    update_statistics = function() {
      private$calculate_cum_stats()
    },
    insert = function(index,
                      t,
                      k,
                      d,
                      action,
                      reward,
                      agent_name,
                      simulation_index,
                      context_value     = NA,
                      theta_value       = NA) {

      if (is.null(action[["propensity"]])) {
        propensity <- NA
      } else {
        propensity <- action[["propensity"]]
      }

      if (is.null(reward[["optimal_reward"]])) {
        optimal_reward <- NA
      } else {
        optimal_reward <- reward[["optimal_reward"]]
      }

      if (is.null(reward[["optimal_arm"]])) {
        optimal_arm <- NA
      } else {
        optimal_arm <- reward[["optimal_arm"]]
      }
      if (!is.vector(context_value)) context_value <- as.vector(context_value)
      if (save_context && !is.null(colnames(context_value))) {   # && !is.null(context_value
        context_value <- context_value[,!colnames(context_value) %in% "(Intercept)"]
      }
      shift_context = 0L
      if (isTRUE(self$save_theta)) {
        theta_value$t      <- t
        theta_value$sim    <- simulation_index
        theta_value$agent  <- agent_name
        theta_value$choice <- action[["choice"]]
        theta_value$reward <- reward[["reward"]]
        theta_value$cum_reward <- reward[["cum_reward"]]
        data.table::set(private$.data, index, 14L, list(list(theta_value)))
        shift_context = 1L
      }
      if (save_context && !is.null(context_value)) {
        if(!isTRUE(self$context_columns_initialized)) {
          private$initialize_data_tables(length(context_value))
          self$context_columns_initialized <- TRUE
        }
        data.table::set(private$.data, index,
                        ((14L+shift_context):(13L+shift_context+length(context_value))),
                        as.list(as.vector(context_value)))
      }
      data.table::set(
        private$.data,
        index,
        1L:13L,
        list(
          t,
          k,
          d,
          simulation_index,
          action[["choice"]],
          reward[["reward"]],
          as.integer(optimal_arm),
          optimal_reward,
          propensity,
          agent_name,
          reward[["regret"]],
          reward[["cum_reward"]],
          reward[["cum_regret"]]
        )
      )
      invisible(self)
    },
    get_agent_list = function() {
      levels(private$.data$agent)
    },
    get_agent_count = function() {
      length(self$get_agent_list())
    },
    get_simulation_count = function() {
      length(levels(as.factor(private$.data$sim)))
    },
    get_arm_choice_percentage = function(limit_agents) {
      private$.data[agent %in% limit_agents][sim != 0][order(choice),
                                                       .(choice = unique(choice),
                                                         percentage = tabulate(choice)/.N)]
    },
    get_meta_data = function() {
      private$.meta
    },
    set_meta_data = function(key, value, group = "sim", agent_name = NULL) {
      upsert <- list()
      upsert[[key]] <- value
      if(!is.null(agent_name)) {
        agent <- list()
        private$.meta[[group]][[key]][[agent_name]] <- NULL
        agent[[agent_name]]    <- append(agent[[agent_name]], upsert)
        private$.meta[[group]] <- append(private$.meta[[group]],agent)
      } else {
        private$.meta[[group]][[key]] <- NULL
        private$.meta[[group]] <- append(private$.meta[[group]],upsert)
      }
    },
    get_cumulative_data = function(limit_agents = NULL, limit_cols = NULL, interval = 1,
                                   cum_average = FALSE) {
      if (is.null(limit_agents)) {
        if (is.null(limit_cols)) {
          private$.cum_stats[t %% interval == 0 | t == 1]
        } else {
          private$.cum_stats[t %% interval == 0 | t == 1, mget(limit_cols)]
        }
      } else {
        if (is.null(limit_cols)) {
          private$.cum_stats[agent %in% limit_agents][t %% interval == 0 | t == 1]
        } else {
          private$.cum_stats[agent %in% limit_agents][t %% interval == 0 | t == 1, mget(limit_cols)]
        }
      }
    },
    get_cumulative_result = function(limit_agents = NULL, as_list = TRUE, limit_cols = NULL, t = NULL) {
      if (is.null(t)) {
        idx <- private$.cum_stats[, .(idx = .I[.N]),   by=agent]$idx
      } else {
        t_int <- as.integer(t)
        idx <- private$.cum_stats[, .(idx = .I[t==t_int]), by=agent]$idx
      }
      cum_results <- private$.cum_stats[idx]
      if (is.null(limit_cols)) {
        if (is.null(limit_agents)) {
          if (as_list) {
            private$data_table_to_named_nested_list(cum_results, transpose = FALSE)
          } else {
            cum_results
          }
        } else {
          if (as_list) {
            private$data_table_to_named_nested_list(cum_results[agent %in% limit_agents], transpose = FALSE)
          } else {
            cum_results[agent %in% limit_agents]
          }
        }
      } else {
        if (is.null(limit_agents)) {
          if (as_list) {
            private$data_table_to_named_nested_list(cum_results[, mget(limit_cols)], transpose = FALSE)
          } else {
            cum_results[, mget(limit_cols)]
          }
        } else {
          if (as_list) {
            private$data_table_to_named_nested_list(cum_results[, mget(limit_cols)]
                                                    [agent %in% limit_agents], transpose = FALSE)
          } else {
            cum_results[, mget(limit_cols)][agent %in% limit_agents]
          }
        }
      }
    },
    save = function(filename = NA) {
      if (is.na(filename)) {
        filename <- paste("contextual_data_",
                          format(Sys.time(), "%y%m%d_%H%M%S"),
                          ".RData",
                          sep = ""
        )
      }
      attr(private$.data, "meta") <- private$.meta
      saveRDS(private$.data, file = filename, compress = TRUE)
      invisible(self)
    },
    load = function(filename, interval = 0, auto_stats = TRUE, bind_to_existing = FALSE) {
      if (isTRUE(bind_to_existing) && nrow(private$.data) > 1 && private$.data$agent[[1]] != "") {
        temp_data <- readRDS(filename)
        if (interval > 0) temp_data <- temp_data[t %% interval == 0]
        private$.data <- rbind(private$.data, temp_data)
        temp_data <- NULL
      } else {
        private$.data <- readRDS(filename)
        if (interval > 0) private$.data <- private$.data[t %% interval == 0]
      }
      private$.meta <- attributes(private$.data)$meta
      if ("opimal" %in% colnames(private$.data))
        setnames(private$.data, old = "opimal", new = "optimal_reward")
      if (isTRUE(auto_stats)) private$calculate_cum_stats()
      invisible(self)
    },
    save_csv = function(filename = NA) {
      if (is.na(filename)) {
        filename <- paste("contextual_data_",
                          format(Sys.time(), "%y%m%d_%H%M%S"),
                          ".csv",
                          sep = ""
        )
      }
      if ("theta" %in% names(private$.data)) {
        fwrite(private$.data[,which(private$.data[,colSums(is.na(private$.data))<nrow(private$.data)]),
                             with =FALSE][, !"theta", with=FALSE], file = filename)
      } else {
        fwrite(private$.data[,which(private$.data[,colSums(is.na(private$.data))<nrow(private$.data)]),
                             with =FALSE], file = filename)
      }
      invisible(self)
    },
    get_data_frame = function() {
      as.data.frame(private$.data)
    },
    set_data_frame = function(df, auto_stats = TRUE) {
      private$.data <- data.table::as.data.table(df)
      if (isTRUE(auto_stats)) private$calculate_cum_stats()
      invisible(self)
    },
    get_data_table = function(limit_agents = NULL, limit_cols = NULL, limit_context = NULL,
                              interval = 1, no_zero_sim = FALSE) {
      if (is.null(limit_agents)) {
        if (is.null(limit_cols)) {
          private$.data[t %% interval == 0 | t == 1][sim != 0]
        } else {
          private$.data[t %% interval == 0 | t == 1, mget(limit_cols)][sim != 0]
        }
      } else {
        if (is.null(limit_cols)) {
          private$.data[agent %in% limit_agents][t %% interval == 0 | t == 1][sim != 0]
        } else {
          private$.data[agent %in% limit_agents][t %% interval == 0 | t == 1, mget(limit_cols)][sim != 0]
        }
      }
    },
    set_data_table = function(dt, auto_stats = TRUE) {
      private$.data <- dt
      if (isTRUE(auto_stats)) private$calculate_cum_stats()
      invisible(self)
    },
    clear_data_table = function() {
      private$.data <- private$.data[0, ]
      invisible(self)
    },
    truncate = function() {
      min_t_sim <- min(private$.data[,max(t), by = c("agent","sim")]$V1)
      private$.data <- private$.data[t<=min_t_sim]
    },
    get_theta = function(limit_agents, to_numeric_matrix = FALSE){
      # unique parameter names, parameter name plus arm nr
      p_names  <- unique(names(unlist(unlist(private$.data[agent %in% limit_agents][1,]$theta,
                                             recursive = FALSE), recursive = FALSE)))
      # number of parameters in theta
      p_number <- length(p_names)
      theta_data <- matrix(unlist(unlist(private$.data[agent %in% limit_agents]$theta,
                                         recursive = FALSE, use.names = FALSE), recursive = FALSE, use.names = FALSE),
                           ncol = p_number, byrow = TRUE)
      colnames(theta_data) <- c(p_names)
      if(isTRUE(to_numeric_matrix)) {
        theta_data <- apply(theta_data, 2, function(x){as.numeric(unlist(x,use.names=FALSE,recursive=FALSE))})
      } else {
        theta_data <- as.data.table(theta_data)
      }
      return(theta_data)
    },
    save_theta_json = function(filename = "theta.json"){
      jj <- rjson::toJSON(private$.data$theta)
      file <- file(filename)
      writeLines(jj, file)
      close(file)
    }
  ),
  private = list(
    .data            = NULL,
    .meta            = NULL,
    .cum_stats       = NULL,
    initialize_data_tables = function(context_cols = NULL) {
      private$.data <- data.table::data.table(
        t = rep(0L, self$n),
        k = rep(0L, self$n),
        d = rep(0L, self$n),
        sim = rep(0L, self$n),
        choice = rep(0.0, self$n),
        reward = rep(0.0, self$n),
        optimal_arm = rep(0L, self$n),
        optimal_reward = rep(0.0, self$n),
        propensity = rep(0.0, self$n),
        agent = rep("", self$n),
        regret = rep(0.0, self$n),
        cum_reward = rep(0.0, self$n),
        cum_regret = rep(0.0, self$n),
        stringsAsFactors = TRUE
      )
      if (isTRUE(self$save_theta)) private$.data$theta <- rep(list(), self$n)
      if (isTRUE(self$save_context)) {
        if (!is.null(context_cols)) {
          context_cols <- c(paste0("X.", seq_along(1:context_cols)))
          private$.data[, (context_cols) := 0.0]
        }
      }

      # meta data
      private$.meta <- list()

      # cumulative data
      private$.cum_stats <- data.table::data.table()
    },
    calculate_cum_stats = function() {

      self$set_meta_data("min_t",min(private$.data[,max(t), by = c("agent","sim")]$V1))
      self$set_meta_data("max_t",max(private$.data[,max(t), by = c("agent","sim")]$V1))

      self$set_meta_data("agents",min(private$.data[, .(count = data.table::uniqueN(agent))]$count))
      self$set_meta_data("simulations",min(private$.data[, .(count = data.table::uniqueN(sim))]$count))

      if (!"optimal_reward" %in% colnames(private$.data))
        private$.data[, optimal_reward:= NA]

      data.table::setkeyv(private$.data,c("t","agent"))

      private$.cum_stats <- private$.data[, list(


        sims                = length(reward),
        sqrt_sims           = sqrt(length(reward)),

        regret_var          = var(regret),
        regret_sd           = sd(regret),
        regret              = mean(regret),

        reward_var          = var(reward),
        reward_sd           = sd(reward),
        reward              = mean(reward),

        optimal_var         = var(as.numeric(optimal_arm == choice)),
        optimal_sd          = sd(as.numeric(optimal_arm == choice)),
        optimal             = mean(as.numeric(optimal_arm == choice)),

        cum_regret_var      = var(cum_regret),
        cum_regret_sd       = sd(cum_regret),
        cum_regret          = mean(cum_regret),

        cum_reward_var      = var(cum_reward),
        cum_reward_sd       = sd(cum_reward),
        cum_reward          = mean(cum_reward) ), by = list(t, agent)]


      private$.cum_stats[, cum_reward_rate_var := cum_reward_var / t]
      private$.cum_stats[, cum_reward_rate_sd := cum_reward_sd / t]
      private$.cum_stats[, cum_reward_rate := cum_reward / t]

      private$.cum_stats[, cum_regret_rate_var := cum_regret_var / t]
      private$.cum_stats[, cum_regret_rate_sd := cum_regret_sd / t]
      private$.cum_stats[, cum_regret_rate := cum_regret / t]

      qn       <- qnorm(0.975)

      private$.cum_stats[, cum_regret_ci      := cum_regret_sd / sqrt_sims * qn]
      private$.cum_stats[, cum_reward_ci      := cum_reward_sd / sqrt_sims * qn]
      private$.cum_stats[, cum_regret_rate_ci := cum_regret_rate_sd / sqrt_sims * qn]
      private$.cum_stats[, cum_reward_rate_ci := cum_reward_rate_sd / sqrt_sims * qn]
      private$.cum_stats[, regret_ci          := regret_sd / sqrt_sims * qn]
      private$.cum_stats[, reward_ci          := reward_sd / sqrt_sims * qn]

      private$.cum_stats[,sqrt_sims:=NULL]

      private$.data[, cum_reward_rate := cum_reward / t]
      private$.data[, cum_regret_rate := cum_regret / t]

      # move agent column to front
      data.table::setcolorder(private$.cum_stats, c("agent", setdiff(names(private$.cum_stats), "agent")))

    },

    data_table_to_named_nested_list = function(dt, transpose = FALSE) {
      df_m <- as.data.frame(dt)
      rownames(df_m) <- df_m[, 1]
      df_m[, 1] <- NULL
      if (!isTRUE(transpose)) {
        apply((df_m), 1, as.list)
      } else {
        apply(t(df_m), 1, as.list)
      }
    },
    finalize = function() {
      self$clear_data_table()
    }
  ),
  active = list(
    data = function(value) {
      if (missing(value)) {
        private$.data
      } else {
        warning("## history$data is read only", call. = FALSE)
      }
    },
    cumulative = function(value) {
      if (missing(value)) {
        self$get_cumulative_result()
      } else {
        warning("## history$cumulative is read only", call. = FALSE)
      }
    },
    meta = function(value) {
      if (missing(value)) {
        self$get_meta_data()
      } else {
        warning("## history$meta is read only", call. = FALSE)
      }
    }
  )
)



#############################################################################################

# POLICY

#############################################################################################

Policy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  public = list(
    action        = NULL,      # action results (list)
    theta         = NULL,      # policy parameters theta (list)
    theta_to_arms = NULL,      # theta to arms "helper" (list)
    is_oracle     = NULL,      # is policy an oracle? (logical)
    class_name    = "Policy",  # policy name - required (character)
    initialize = function() {
      # Is called before the Policy instance has been cloned.
      self$theta  <- list()    # initializes theta list
      self$action <- list()    # initializes action list
      is_oracle   <- FALSE     # very seldom TRUE
      invisible(self)
    },
    post_initialization = function() {
      # Is called after a Simulator has cloned the Policy instance [number_of_simulations] times.
      # Do sim level random generation here.
      invisible(self)
    },
    set_parameters = function(context_params) {
      # Parameter initialisation happens here.
    },
    get_action = function(t, context) {
      # Selects an arm based on paramters in self$theta and the current context,
      # the index of the chosen arm through action$choice.
      stop("Policy$get_action() has not been implemented.", call. = FALSE)
    },
    set_reward = function(t, context, action, reward) {
      # Updates parameters in theta based on current context and
      # the reward that was awarded by the bandit for the policy's action$choice.
      stop("Policy$set_reward() has not been implemented.", call. = FALSE)
    },
    initialize_theta = function(k) {
      # Called by a policy's agent during contextual's initialization phase.

      # The optional "helper variable" self$theta_to_arms
      # is parsed here. That is, when self$theta_to_arms exists, it is copied
      # self$k times, and each copy is made available through self$theta.
      if (!is.null(self$theta_to_arms)) {
        for (param_index in seq_along(self$theta_to_arms)) {
          self$theta[[ names(self$theta_to_arms)[param_index] ]] <-
            rep(list(self$theta_to_arms[[param_index]]),k)
        }
      }
      self$theta
    }
  )
)

#############################################################################################

# PLOT

#############################################################################################

Plot <- R6::R6Class(
  "Plot",
  public = list(
    history = NULL,

    cumulative = function(history,

                          regret             = TRUE,
                          disp               = NULL,
                          plot_only_disp     = FALSE,
                          rate               = FALSE,
                          interval           = 1,
                          traces             = FALSE,
                          traces_max         = 100,
                          traces_alpha       = 0.3,
                          smooth             = FALSE,
                          no_par             = FALSE,
                          xlim               = NULL,
                          ylim               = NULL,
                          xlab               = NULL,
                          ylab               = NULL,
                          legend             = TRUE,
                          log                = "",
                          use_colors         = TRUE,
                          color_step         = 1,
                          lty_step           = 1,
                          lwd                = 2,
                          legend_labels      = NULL,
                          legend_border      = NULL,
                          legend_position    = "topleft",
                          legend_title       = NULL,
                          limit_agents       = NULL,
                          limit_context      = NULL,
                          trunc_over_agents  = TRUE,
                          trunc_per_agent    = TRUE) {

      self$history       <- history

      if (regret) {
        if (rate) {
          ylab_title     <- "Cumulative regret rate"
          line_data_name <- "cum_regret_rate"
          disp_data_name <- "cum_regret_rate_none"
        } else {
          ylab_title     <- "Cumulative regret"
          line_data_name <- "cum_regret"
          disp_data_name <- "cum_regret_none"
        }
      } else {
        if (rate) {
          ylab_title     <- "Cumulative reward rate"
          line_data_name <- "cum_reward_rate"
          disp_data_name <- "cum_reward_rate_none"
        } else {
          ylab_title     <- "Cumulative reward"
          line_data_name <- "cum_reward"
          disp_data_name <- "cum_reward_none"
        }
      }

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        xlab                = xlab,
        ylab                = ylab,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        rate                = rate,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },

    optimal = function(history,
                       disp               = NULL,
                       plot_only_disp     = FALSE,
                       rate               = FALSE,
                       interval           = 1,
                       traces             = FALSE,
                       traces_max         = 100,
                       traces_alpha       = 0.3,
                       smooth             = FALSE,
                       no_par             = FALSE,
                       xlim               = NULL,
                       ylim               = NULL,
                       xlab               = NULL,
                       ylab               = NULL,
                       legend             = TRUE,
                       use_colors         = TRUE,
                       log                = "",
                       color_step         = 1,
                       lty_step           = 1,
                       lwd                = 2,
                       legend_labels      = NULL,
                       legend_border      = NULL,
                       legend_position    = "topleft",
                       legend_title       = NULL,
                       limit_agents       = NULL,
                       limit_context      = NULL,
                       trunc_over_agents  = TRUE,
                       trunc_per_agent    = TRUE) {

      self$history <- history

      ylab_title     <- "Optimal action"
      line_data_name <- "optimal"
      disp_data_name   <- "optimal_none"

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },


    average = function(history,
                       regret             = TRUE,
                       disp               = NULL,
                       plot_only_disp     = FALSE,
                       rate               = FALSE,
                       interval           = 1,
                       traces             = FALSE,
                       traces_max         = 100,
                       traces_alpha       = 0.3,
                       smooth             = FALSE,
                       no_par             = FALSE,
                       xlim               = NULL,
                       ylim               = NULL,
                       xlab               = NULL,
                       ylab               = NULL,
                       legend             = TRUE,
                       use_colors         = TRUE,
                       log                = "",
                       color_step         = 1,
                       lty_step           = 1,
                       lwd                = 2,
                       cum_average        = FALSE,
                       legend_labels      = NULL,
                       legend_border      = NULL,
                       legend_position    = "topleft",
                       legend_title       = NULL,
                       limit_agents       = NULL,
                       limit_context      = NULL,
                       trunc_over_agents  = TRUE,
                       trunc_per_agent    = TRUE) {
      self$history <- history

      if (regret) {
        ylab_title     <- "Average regret"
        line_data_name <- "regret"
        disp_data_name   <- "regret_none"
      } else {
        ylab_title     <- "Average reward"
        line_data_name <- "reward"
        disp_data_name   <- "reward_none"
      }

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        xlab                = xlab,
        ylab                = ylab,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        cum_average         = cum_average,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        rate                = rate,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },

    arms = function(history,

                    no_par             = FALSE,
                    legend             = TRUE,
                    use_colors         = TRUE,
                    log                = "",
                    interval           = 1,
                    xlim               = NULL,
                    ylim               = NULL,
                    xlab               = NULL,
                    ylab               = NULL,
                    legend_labels      = NULL,
                    legend_border      = NULL,
                    legend_position    = "bottomright",
                    legend_title       = NULL,
                    limit_context      = NULL,
                    smooth             = FALSE,
                    trunc_over_agents  = TRUE,
                    limit_agents       = NULL) {

      self$history <- history

      if (!isTRUE(no_par)) {
        dev.hold()
        old.par <- par(no.readonly = TRUE)
        par(mar = c(5, 5, 1, 1))
      }

      if(!is.null(limit_context)) {
        dt <- self$history$get_data_table(
          limit_cols   = c("agent", "t", "choice", "sim", limit_context),
          limit_agents = limit_agents,
          interval     = interval
        )
        dt <- dt[dt[, Reduce(`|`, lapply(.SD, `==`, 1)),.SDcols = limit_context],]
      } else {
        dt <- self$history$get_data_table(
          limit_cols   = c("agent", "t", "choice", "sim"),
          limit_agents = limit_agents,
          interval     = interval
        )
      }

      if(isTRUE(trunc_over_agents))  {
        min_t_sim <- min(dt[,max(t), by = c("agent","sim")]$V1)
        dt <- dt[t<=min_t_sim]
      }

      ylab_title        <- "Arm choice %"
      agent_levels      <- levels(droplevels(dt$agent))

      if (length(agent_levels) > 1) {
        warning(strwrap(
          prefix = " ", initial = "",
          "## Arm percentage plot always plots the results of one agent, either at
          index position one, or the first agent specified in limit_agents."
        ),
        call. = FALSE
        )
      }

      dt                <- dt[agent == agent_levels[1]]

      dt$agent          <- NULL
      data.table::setkey(dt, t, choice)
      data              <- dt[data.table::CJ(t, choice, unique = TRUE), list(arm_count =  .N), by = .EACHI]

      #data              <- dt[, list(arm_count =  .N), by = list(t, choice)]

      max_sim           <- dt[, max(sim)]
      max_t             <- dt[, max(t)]

      arm_levels        <- levels(as.factor(data$choice))
      max_arm           <- length(arm_levels)
      N                 <- dt[,.N,by=c("t")]$N
      N                 <- rep(N,each=max_arm)

      data$arm_count    <- as.double((unlist(data$arm_count, FALSE, FALSE) / N) * 100L)

      eg                <- expand.grid(t = dt[sim == 1]$t, choice = seq(1.0, max_arm, 1))
      data              <- merge(data, eg, all = TRUE)
      # turn NA into 0
      for (j in seq_len(ncol(data)))
        set(data,which(is.na(data[[j]])),j,0)

      data$dataum       <- ave(data$arm_count, data$t, FUN = cumsum)
      data$zero         <- 0.0
      min_ylim          <- 0
      max_ylim          <- 100

      if (isTRUE(smooth)) {
        for (arm in arm_levels) {
          data[data$choice == arm, c("t", "dataum") :=
                 supsmu(data[data$choice == arm]$t, data[data$choice == arm]$dataum, bass = 9)]
        }
      }

      data.table::setorder(data, choice, t)
      plot.new()

      if (!is.null(xlim)) {
        min_xlim <- xlim[1]
        max_xlim <- xlim[2]
      } else {
        min_xlim <- 1
        max_xlim <- data[, max(t)]
      }
      if (!is.null(ylim)) {
        min_ylim <- ylim[1]
        max_ylim <- ylim[2]
      }
      plot.window(
        xlim = c(min_xlim, max_xlim),
        ylim = c(min_ylim, max_ylim)
      )

      if (isTRUE(use_colors)) {
        cl <- private$gg_color_hue(length(arm_levels))
      } else {
        cl <- gray.colors(length(arm_levels))
      }

      color <- 1
      polygon(
        c(data[data$choice == 1]$t, rev(data[data$choice == 1]$t)),
        c(data[data$choice == 1]$dataum, rev(data[data$choice == 1]$zero)),
        col = adjustcolor(cl[color], alpha.f = 0.6),
        border = NA
      )

      color <- 2
      for (arm_nr in c(2:length(arm_levels))) {
        polygon(
          c(data[data$choice == arm_nr]$t, rev(data[data$choice == arm_nr]$t)),
          c(data[data$choice == arm_nr - 1]$dataum, rev(data[data$choice == arm_nr]$dataum)),
          col = adjustcolor(cl[color], alpha.f = 0.6),
          border = NA
        )
        color <- color + 1
      }

      if (is.null(legend_title)) {
        legend_title <- agent_levels[1]
        if(!is.null(limit_context))
          legend_title <- paste(legend_title,limit_context)
      }

      if (is.null(legend_position)) {
        legend_position <- "bottomright"
      }

      if (!is.null(legend_labels)) {
        legend_labels <- legend_labels
      } else {
        legend_labels <- paste("arm", arm_levels, sep = " ")
      }

      axis(1)
      axis(2)
      title(xlab = "Time Step")
      title(ylab = ylab_title)
      box()
      if (legend) {
        legend(
          legend_position,
          NULL,
          legend_labels,
          col = adjustcolor(cl, alpha.f = 0.6),
          title = legend_title,
          pch = 15,
          pt.cex = 1.2,
          bg = "white",
          inset = c(0.08, 0.1)
        )
      }
      if (!isTRUE(no_par)) {
        dev.flush()
        par(old.par)
      }
      invisible(recordPlot())
    }
  ),
  private = list(
    cum_average = function(cx) {
      cx <- c(0,cx)
      cx[(2):length(cx)] - cx[1:(length(cx) - 1)]
    },
    do_plot = function(line_data_name      = line_data_name,
                       disp_data_name      = disp_data_name,
                       disp                = NULL,
                       plot_only_disp      = FALSE,
                       ylab_title          = NULL,
                       use_colors          = FALSE,
                       log                 = "",
                       legend              = TRUE,
                       no_par              = FALSE,
                       xlim                = NULL,
                       ylim                = NULL,
                       xlab                = NULL,
                       ylab                = NULL,
                       interval            = 1,
                       color_step          = 1,
                       lty_step            = 1,
                       lwd                 = 2,
                       legend_labels       = NULL,
                       legend_border       = NULL,
                       legend_position     = "topleft",
                       legend_title        = NULL,
                       limit_agents        = NULL,
                       limit_context       = NULL,
                       traces              = NULL,
                       traces_max          = 100,
                       traces_alpha        = 0.3,
                       cum_average         = FALSE,
                       smooth              = FALSE,
                       rate                = FALSE,
                       trunc_over_agents   = TRUE,
                       trunc_per_agent     = TRUE) {

      cum_flip <- FALSE
      if((line_data_name=="reward" || line_data_name=="regret") && isTRUE(cum_average)) {
        line_data_name <- paste0("cum_",line_data_name)
        cum_flip = TRUE
      }

      if (interval==1 && as.integer(self$history$meta$sim$max_t) > 1850) {
        interval <- ceiling(as.integer(self$history$meta$sim$max_t)/1850) # nocov
        if(isTRUE(cum_average) && isTRUE(cum_flip))  {
          warning(strwrap(
            prefix = " ", initial = "",
            paste0("## As cum_reward was set to TRUE while plotting more than 1850 time steps,
            the reward plot has been smoothed automatically using a window length of ",interval,
                   " timesteps.")
          ),
          call. = FALSE
          )
        }
      }

      if (!is.null(disp) && disp %in% c("sd", "var", "ci")) {

        disp_data_name <- gsub("none", disp, disp_data_name)
        data <-
          self$history$get_cumulative_data(
            limit_cols   = c("agent", "t", "sims", line_data_name, disp_data_name),
            limit_agents = limit_agents,
            interval     = interval
          )

      } else {
        disp <- NULL
        data <-
          self$history$get_cumulative_data(
            limit_cols   = c("agent", "t", "sims", line_data_name),
            limit_agents = limit_agents,
            interval     = interval
          )
      }

      agent_levels <- levels(droplevels(data$agent))
      n_agents <- length(agent_levels)

      # turn NA into 0
      for (j in seq_len(ncol(data)))
        data.table::set(data,which(is.na(data[[j]])),j,0)

      if(isTRUE(trunc_per_agent))  {
        data <- data[data$sims == max(data$sims)]
      }

      if(isTRUE(trunc_over_agents))  {
        min_t_sim <- min(data[,max(t), by = c("agent")]$V1)
        data <- data[t<=min_t_sim]
      }

      if (!is.null(xlim)) {
        min_xlim <- xlim[1]
        max_xlim <- xlim[2]
      } else {
        min_xlim <- 1
        max_xlim <- data[, max(t)]
      }

      data.table::setorder(data, agent, t)

      if(cum_flip==TRUE) {
        if (line_data_name == "cum_reward") {
          line_data_name <- "reward"
          for (agent_name in agent_levels) {
            data[data$agent == agent_name,
                 reward := private$cum_average(data[data$agent == agent_name]$cum_reward)/interval]
          }
        } else {
          line_data_name <- "cum_regret"
          for (agent_name in agent_levels) {
            data[data$agent == agent_name,
                 regret := private$cum_average(data[data$agent == agent_name]$cum_regret)/interval]
          }
        }
      }

      if(!is.null(xlim)) data <- data[t>=xlim[1] & t<=xlim[2]]

      if(!is.null(limit_context)) {
        data <- data[data[, Reduce(`|`, lapply(.SD, `==`, 1)),.SDcols = sel],]
      }

      data.table::setorder(data, agent, t)

      if (isTRUE(smooth)) {
        for (agent_name in agent_levels) {
          data[data$agent == agent_name, c("t", line_data_name) :=
                 supsmu(data[data$agent == agent_name]$t, data[data$agent == agent_name][[line_data_name]])]
          if (!is.null(disp)) {
            data[data$agent == agent_name, c("t", disp_data_name) :=
                   supsmu(data[data$agent == agent_name]$t, data[data$agent == agent_name][[disp_data_name]])]
          }
        }
      }

      if (!isTRUE(no_par)) {
        dev.hold()
        old.par <- par(no.readonly = TRUE)
        par(mar = c(5, 5, 1, 1))
      }

      if (!is.null(disp) && !isTRUE(plot_only_disp)) {
        disp_range <- data[[line_data_name]] + outer(data[[disp_data_name]], c(1, -1))
        data     <- cbind(data, disp_range)
        colnames(data)[colnames(data) == "V2"] <- "disp_lower"
        colnames(data)[colnames(data) == "V1"] <- "disp_upper"
      }

      if (isTRUE(plot_only_disp)) {
        if(is.null(disp)) stop("Need to set disp to 'var','sd' or 'ci' when plot_only_disp is TRUE",
                               call. = FALSE)
        line_data_name = disp_data_name
      }

      plot.new()
      cl <- private$gg_color_hue(round(n_agents / color_step))
      cl <- rep(cl, round(color_step))

      if (lty_step > 1) {
        lt <- rep(1:round(lty_step), each = round(n_agents / lty_step))
      } else {
        lt <- rep(1, n_agents)
      }
      if (!isTRUE(use_colors) && lty_step == 1) {
        lty_step <- n_agents
        lt <- rep(1:round(lty_step), each = round(n_agents / lty_step))
      }

      if (!is.null(disp) && !isTRUE(plot_only_disp) &&
          !is.na(data[, min(disp_lower)]) && !is.na(data[, min(disp_upper)])) {
        min_ylim <- data[, min(disp_lower)]
        max_ylim <- data[, max(disp_upper)]
      } else {
        min_ylim <- data[, min(data[[line_data_name]])]
        max_ylim <- data[, max(data[[line_data_name]])]
      }


      if (!is.null(ylim)) {
        min_ylim <- ylim[1]
        max_ylim <- ylim[2]
      }
      plot.window(
        xlim = c(min_xlim, max_xlim),
        ylim = c(min_ylim, max_ylim),
        log = log
      )

      if (isTRUE(traces) && !isTRUE(plot_only_disp)) {
        dt <- self$history$get_data_table(limit_agents = limit_agents, interval = interval)
        data.table::setorder(dt, agent, sim, t)
        for (agent_name in agent_levels) {
          agent_sims <- unique(dt[dt$agent == agent_name]$sim)
          for (as in head(agent_sims, traces_max)) {
            Sys.sleep(0)
            if (isTRUE(smooth)) {
              lines(supsmu(
                dt[dt$agent == agent_name & dt$sim == as]$t,
                dt[dt$agent == agent_name & dt$sim == as][[line_data_name]]
              ),
              lwd = lwd,
              col = rgb(0.8, 0.8, 0.8, traces_alpha)
              )
            } else {
              lines(dt[dt$agent == agent_name & dt$sim == as]$t,
                    dt[dt$agent == agent_name &
                         dt$sim == as][[line_data_name]],
                    lwd = lwd,
                    col = rgb(0.8, 0.8, 0.8, traces_alpha)
              )
            }
          }
        }
      }

      if (isTRUE(use_colors)) {
        if (!is.null(disp) && !isTRUE(plot_only_disp)) {
          color <- 1
          for (agent_name in agent_levels) {
            polygon(
              c(data[data$agent == agent_name]$t, rev(data[data$agent == agent_name]$t)),
              c(data[data$agent == agent_name]$disp_lower, rev(data[data$agent == agent_name]$disp_upper)),
              col = adjustcolor(cl[color], alpha.f = 0.3),
              border = NA
            )
            color <- color + 1
          }
        }
        line_counter <- 1
        for (agent_name in agent_levels) {
          lines(
            data[data$agent == agent_name]$t,
            data[data$agent == agent_name][[line_data_name]],
            lwd  = lwd,
            lty  = lt[line_counter],
            col  = adjustcolor(cl[line_counter], alpha.f = 0.9),
            type = "l"
          )
          line_counter <- line_counter + 1
        }
      } else {
        line_counter <- 1
        for (agent_name in agent_levels) {
          if (!is.null(disp) && !isTRUE(plot_only_disp)) {
            polygon(
              c(data[data$agent == agent_name]$t, rev(data[data$agent == agent_name]$t)),
              c(data[data$agent == agent_name]$disp_lower, rev(data[data$agent == agent_name]$disp_upper)),
              col = rgb(0.8, 0.8, 0.8, 0.4),
              border = NA
            )
          }
          lines(
            data[data$agent == agent_name]$t,
            data[data$agent == agent_name][[line_data_name]],
            lwd = lwd,
            lty = lt[line_counter],
            col = rgb(0.2, 0.2, 0.2, 0.8),
            type = "l"
          )
          line_counter <- line_counter + 1
        }
      }
      axis(1)
      axis(2)
      if (is.null(xlab)) xlab = "Time step"
      title(xlab = xlab)
      if(isTRUE(plot_only_disp)) ylab_title <- paste0(ylab_title,": ",disp)
      if (is.null(ylab)) ylab = ylab_title
      title(ylab = ylab)
      box()
      if (legend) {
        if (!is.null(legend_labels)) {
          agent_levels <- legend_labels
        }
        if (!is.null(legend_border)) {
          bty <- "n"
        } else {
          bty <- "o"
        }
        if (!isTRUE(use_colors)) {
          cl <- rgb(0.2, 0.2, 0.2, 0.8)
        }
        legend(
          legend_position,
          NULL,
          agent_levels,
          col   = cl,
          title = legend_title,
          lwd   = lwd,
          lty   = lt,
          bty   = bty,
          bg    = "white"
        )
      }
      if (!isTRUE(no_par)) {
        dev.flush()
        par(old.par)
      }
    },
    gg_color_hue = function(n) {
      hues <- seq(15, 375, length = n + 1)
      hcl(h = hues, l = 65, c = 100)[1:n]
    }
  )
)


#' @importFrom dplyr filter
#' @importFrom magrittr %>%
#' @importFrom stats runif

# EXTRACT 2D FROm 3D ------------------------------------------------------------------------

extract_2d_from_3d <- function(array3d, depth_indices) {
  # Get array dimensions
  dims <- dim(array3d)
  nrow <- dims[1]  # Rows
  ncol <- dims[2]  # Columns

  # Ensure depth_indices length matches required rows
  if (length(depth_indices) != nrow) {
    stop("The arm selection vector should have same length as the first dimension of the policy array.")
  }

  # Vectorized index calculation
  i <- rep(1:nrow, each = ncol)  # Row indices
  j <- rep(1:ncol, times = nrow) # Column indices
  k <- rep(depth_indices, each = ncol)  # Depth indices

  # Calculate linear indices for efficient extraction
  linear_indices <- i + (j - 1) * nrow + (k - 1) * nrow * ncol

  # Create result matrix using vectorized indexing
  result_matrix <- matrix(array3d[linear_indices], nrow = nrow, ncol = ncol, byrow = TRUE)

  return(result_matrix)
}

# COMPUTE PROBA, to be applied on each agent, sim subgroup --------------------------------------------

compute_probas <- function(df, policy, policy_name, batch_size) {
  # Extract contexts and arms for the entire (agent, sim) group
  contexts <- df$context
  ind_arm <- df$choice
  theta_all <- df[, theta]

  # Use A_inv if LinTSPolicy is in the string of policy_name
  key_A <- ifelse(grepl("LinTSPolicy", policy_name), "A_inv", "A")
  key_b <- "b"

  # Extract A and b
  A_list <- lapply(theta_all, `[[`, key_A)
  b_list <- lapply(theta_all, `[[`, key_b)

  # Subset A and b for batch processing
  if (batch_size > 1) {
    indices_to_keep <- seq(batch_size, length(A_list), by = batch_size)
    A_list <- A_list[indices_to_keep]
    b_list <- b_list[indices_to_keep]
  }

  # Compute the probability matrix based on the policy name
  probas_matrix <- switch(
    policy_name,
    "ContextualEpsilonGreedyPolicy" =,
    "BatchContextualEpsilonGreedyPolicy" = get_proba_c_eps_greedy(policy$epsilon, A_list, b_list, contexts, ind_arm, batch_size),

    "ContextualLinTSPolicy" =,
    "BatchContextualLinTSPolicy" = get_proba_thompson(policy$sigma, A_list, b_list, contexts, ind_arm, batch_size),

    "LinUCBDisjointPolicyEpsilon" =,
    "BatchLinUCBDisjointPolicyEpsilon" = get_proba_ucb_disjoint(policy$alpha, policy$epsilon, A_list, b_list, contexts, ind_arm, batch_size),

    stop("Unsupported policy_name: Choose among ContextualEpsilonGreedyPolicy, BatchContextualEpsilonGreedyPolicy,
         ContextualLinTSPolicy, BatchContextualLinTSPolicy, LinUCBDisjointPolicyEpsilon, BatchLinUCBDisjointPolicyEpsilon")
  )

  # Store each column of probas_matrix in a list
  # i.e. each element of the list corresponds to one context (and is a vector of proba across proba param)
  # List of T vectors of length nb_batch

  probas_list <- split(probas_matrix, row(probas_matrix))

  # Return df with probabilities for each row
  return(probas_list)
}



# GET PROBA EPSILON GREEDY -------------------------------------------------------------------------

get_proba_c_eps_greedy <- function(eps = 0.1, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch of matrices (T, K): for each policy, expected reward across arms given all contexts
  # a policy is represented by a (d, K): K vectors theta = A^-1 b of shape (d x 1)
  # we then multiply by contexts to get a (T, d) x (d, K) = (T, K)
  expected_rewards <- lapply(seq_len(nb_batch), function(t) {
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat
  }) # (T, K)

  # Convert expected_rewards (list of nb_batch matrices) into a 3D array (T × K × nb_batch)
  # T x K x nb_batch = context x arm x policy
  expected_rewards_array <- simplify2array(expected_rewards)

  # Swap last dimension (nb_batch) with second dimension (K) -> (T × nb_batch × K)
  expected_rewards_array <- aperm(expected_rewards_array, c(1, 3, 2))

  # Find max expected rewards across K for each (T, nb_batch) combo
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × nb_batch)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_batch, K))

  # For each (T, nb_batch) combo, says if arm had max expected reward or not (1 or 0)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × nb_batch × K)

  # For each (T, nb_batch) combo, count the number of best arms
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × nb_batch)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  # i.e. whether the arm chosen in the history had max expected reward or not
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × nb_batch)

  # Compute final probabilities (T × nb_batch)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

  return(proba_results)  # Returns (T × nb_batch) matrix of probabilities, one context per row
}

get_proba_c_eps_greedy_penultimate <- function(eps = 0.1, A_list, b_list, context_matrix) {
  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # Theta hat matrix of shape (d, K)
  theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[k]], b_list[[k]]), simplify = "matrix")

  # Expected rewards matrix of shape (B, K)
  expected_rewards <-  context_matrix  %*% theta_hat

  # Find max expected rewards for each row in every B
  max_rewards <- apply(expected_rewards, 1, max)  # Shape: (B)

  max_rewards_expanded <- array(max_rewards, dim = c(B, K))

  # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards == max_rewards_expanded  # Shape: (B × K)

  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, 1, sum)  # Shape: (B)

  # Compute final probabilities (B × K)
  proba_results <- (1 - eps) * ties / num_best_arms + eps / K

  return(proba_results)  # Returns (B × K) matrix of probabilities, one context per row
}

# GET PROBA UCB DISJOINT WITH EPSILON ---------------------------------------------------------

get_proba_ucb_disjoint <- function(alpha=1.0, eps = 0.1, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch of matrices (T, K): for each policy, expected reward across arms across all contexts
  # a policy is represented by a (d, K): K vectors theta = A^-1 b of shape (d x 1)
  # we then multiply by contexts to get a (T, d) x (d, K) = (T, K)
  mu <- lapply(seq_len(nb_batch), function(t) {
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat # (T x K)
  }) # (T, K)

  # List of length nb_batch of matrices (T, K): for each policy, standard deviation of expected reward
  # across arms across all contexts
  variance <- lapply(seq_len(nb_batch), function(t) {
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% inv(A_list[[t]][[k]]) # (T x d)
      # We have to do that NOT to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance_terms <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
      sqrt(variance_terms)
    }, simplify = "matrix") # (T x K)
  })

  # Convert mu and variance (list of nb_batch matrices) into 3D arrays (T × K × nb_batch)
  # T x K x nb_batch = context x arm x policy
  mu_array <- simplify2array(mu)
  variance_array <- simplify2array(variance)

  # Swap last dimension (nb_batch) with second dimension (K) -> (T × nb_batch × K)
  # T x nb_batch x K = context x policy x arm
  mu_array <- aperm(mu_array, c(1, 3, 2))
  variance_array <- aperm(variance_array, c(1, 3, 2))

  expected_rewards_array <- mu_array + alpha * variance_array

  # Find max expected rewards across K for each (T, nb_batch) combo
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × nb_batch)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_batch, K))

  # For each (T, nb_batch) combo, says if arm had max expected reward or not (1 or 0)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × nb_batch × K)

  # For each (T, nb_batch) combo, count the number of best arms
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × nb_batch)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  # i.e. whether the arm chosen in the history had max expected reward or not
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × nb_batch)

  # Compute final probabilities (T × nb_batch)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

  return(proba_results)
}

get_proba_ucb_disjoint_penultimate <- function(alpha=1.0, eps = 0.1, A_list, b_list, context_matrix) {

  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # Theta hat matrix of shape (d, K)
  theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[k]], b_list[[k]]), simplify = "matrix")

  # Expected rewards matrix of shape (B, K)
  mu <-  context_matrix  %*% theta_hat

  variance_matrix <- sapply(seq_len(K), function (k) {
    semi_var <- context_matrix %*% inv(A_list[[k]]) # (B x d)
    # We have to do that NOT to end up with Xi * A_inv * t(Xj) for all combinations of i,j
    # We only want the combinations where i = j
    variance_terms <- rowSums(semi_var * context_matrix) # (vector of length B for each k)
    # for a given policy, for a given arm, we have T sigmas: one per context
    sqrt(variance_terms)
  }, simplify = "matrix") # (B x K)

  expected_rewards <- mu + alpha * variance_matrix

  # Find max expected rewards for each row in every B
  max_rewards <- apply(expected_rewards, 1, max)  # Shape: (B)

  max_rewards_expanded <- array(max_rewards, dim = c(B, K))

  # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards == max_rewards_expanded  # Shape: (B × K)

  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, 1, sum)  # Shape: (B)

  # Compute final probabilities (B × K)
  proba_results <- (1 - eps) * ties / num_best_arms + eps / K

  return(proba_results)

}

# GET PROBA THOMPSON SAMPLING ---------------------------------------------------------------------

get_proba_thompson <- function(sigma = 0.01, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch giving for each policy t, the array of probabilities under each context j
  # of selecting Aj
  result <- lapply(seq_len(nb_batch), function(t) {

    # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
    theta_hat <- sapply(seq_len(K), function(k) A_list[[t]][[k]] %*% b_list[[t]][[k]], simplify = "matrix")

    mean <- context_matrix  %*% theta_hat # (T x K)
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% (sigma * A_list[[t]][[k]]) # (T x d)
      # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
    }, simplify = "matrix") # (T x K)

    proba_results <- numeric(nb_timesteps)

    for (j in 1:nb_timesteps) {

      mean_k <- mean[j, ind_arm[j]]
      var_k  <-  variance_matrix[j, ind_arm[j]]
      # var_k <- max(var_k, 1e-6)

      competing_arms <- setdiff(1:K, ind_arm[j])

      mean_values <- mean[j,competing_arms]
      var_values <- variance_matrix[j, competing_arms]
      # var_values <- pmax(var_values, 1e-6)

      # Define the function for integration
      integrand <- function(x) {
        log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

        for (i in seq_along(mean_values)) {
          log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[i], sd = sqrt(var_values[i]), log.p = TRUE)
        }

        return(exp(log_p_xk))  # Convert back to probability space
      }

      # lower_bound <- mean_k - 3 * sqrt(var_k)
      # upper_bound <- mean_k + 3 * sqrt(var_k)
      all_means <- c(mean_k, mean_values)
      all_vars <- c(var_k, var_values)
      lower_bound <- min(all_means - 3 * sqrt(all_vars))
      upper_bound <- max(all_means + 3 * sqrt(all_vars))

      # Adaptive numerical integration
      prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 10, abs.tol = 1e-2)$value

      clip <- 1e-3

      proba_results[j] <- pmax(clip, pmin(prob, 1-clip))
    }

    return(proba_results)
  })

  # result is a list giving for each policy t, the array of probabilities under each context j
  # of selecting Aj
  result_matrix <- simplify2array(result) # a row should be a context, policies in columns

  return(result_matrix)
}

get_proba_thompson_penultimate <- function(sigma = 0.01, A_list, b_list, context_matrix) {

  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # For penultimate policy, gives the array of probabilities under each context j (1:B)
  # of selecting arm k (1:K)

  # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
  theta_hat <- sapply(seq_len(K), function(k) A_list[[k]] %*% b_list[[k]], simplify = "matrix")

  mean <- context_matrix  %*% theta_hat # (B x K)
  variance_matrix <- sapply(seq_len(K), function (k) {
    semi_var <- context_matrix %*% (sigma * A_list[[k]]) # (B x d)
    # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
    # We only want the combinations where i = j
    variance <- rowSums(semi_var * context_matrix) # (vector of length B for each k)
    # for a given policy, for a given arm, we have T sigmas: one per context
  }, simplify = "matrix") # (B x K)


  result <- vector("list", K)

  for (k in 1:K) {

    proba_results <- numeric(B)

    for (j in 1:B) {

      mean_k <- mean[j, k]
      var_k  <-  variance_matrix[j, k]
      #var_k <- max(var_k, 1e-6)

      competing_arms <- setdiff(1:K, k)

      mean_values <- mean[j,competing_arms]
      var_values <- variance_matrix[j, competing_arms]
      #var_values <- pmax(var_values, 1e-6)

      # Define the function for integration
      integrand <- function(x) {
        log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

        for (i in seq_along(mean_values)) {
          log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[i], sd = sqrt(var_values[i]), log.p = TRUE)
        }

        return(exp(log_p_xk))  # Convert back to probability space
      }

      # lower_bound <- mean_k - 3 * sqrt(var_k)
      # upper_bound <- mean_k + 3 * sqrt(var_k)
      all_means <- c(mean_k, mean_values)
      all_vars <- c(var_k, var_values)
      lower_bound <- min(all_means - 3 * sqrt(all_vars))
      upper_bound <- max(all_means + 3 * sqrt(all_vars))


      # Adaptive numerical integration
      prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 10, abs.tol = 1e-2)$value

      clip <- 1e-3

      proba_results[j] <- pmax(clip, pmin(prob, 1-clip))
    }

    result[[k]] <- proba_results
  }

  # result is a list giving for each arm k, the array of probabilities under each context j
  # of selecting arm k
  result_matrix <- do.call(cbind, result)
  # result_matrix <- simplify2array(result) # a row should be a context, arms in columns (B x K)

  return(result_matrix)
}


# COMPUTE ESTIMAND ------------------------------------------------------------------

compute_estimand <- function(sim_data, list_betas, policy, policy_name, batch_size, bandit) {

  # GET PARAMS OF PI_{T-1} (or PI_{T-batch_size} more generally) ------------------------
  last_timestep <- max(sim_data$t)

  last_row <- sim_data %>% filter(t == last_timestep - batch_size)

  theta_info <- last_row$theta[[1]]  # Extract the actual theta list (removing outer list structure)

  # Use A_inv if LinTSPolicy is in the string of policy_name
  key_A <- ifelse(grepl("LinTSPolicy", policy_name), "A_inv", "A")
  key_b <- "b"

  A_list <- theta_info[[key_A]]
  b_list <- theta_info[[key_b]]

  # GET BETA MATRIX FOR CURRENT SIM --------------------------------------------------

  # Safely extract simulation index
  sim_index <- theta_info$sim - 1

  beta_matrix <- list_betas[[sim_index]]  # Shape (features x arms)

  # GET INDEPENDENT CONTEXTS FROM OTHER SIMs ---------------------------------------
  B <- 1000
  d <- bandit$d
  context_matrix <- matrix(rnorm(B * d), nrow = B, ncol = d)

  # # # Take a random subset of 1000 records (if available)
  # num_samples <- min(1000, nrow(context_matrix))  # Ensure we don’t sample more than available
  # context_matrix <- context_matrix[sample(nrow(context_matrix), num_samples, replace = FALSE), , drop = FALSE]  # Shape (1000 × d)

  # Compute true linear rewards via matrix multiplication
  # True linear rewards (B × K) = (B × d) * (d × K)
  true_linear_rewards <- context_matrix %*% beta_matrix  # Shape (B x K)

  # Compute the probability matrix based on the policy name
  policy_probs <- switch(
    policy_name,
    "ContextualEpsilonGreedyPolicy" =,
    "BatchContextualEpsilonGreedyPolicy" = get_proba_c_eps_greedy_penultimate(policy$epsilon, A_list, b_list, context_matrix),  # Should be (B x K)

    "ContextualLinTSPolicy" =,
    "BatchContextualLinTSPolicy" = get_proba_thompson_penultimate(policy$sigma, A_list, b_list, context_matrix),

    "LinUCBDisjointPolicyEpsilon" =,
    "BatchLinUCBDisjointPolicyEpsilon" = get_proba_ucb_disjoint_penultimate(policy$alpha, policy$epsilon, A_list, b_list, context_matrix),

    stop("Unsupported policy_name: Choose among ContextualEpsilonGreedyPolicy, BatchContextualEpsilonGreedyPolicy,
         ContextualLinTSPolicy, BatchContextualLinTSPolicy, LinUCBDisjointPolicyEpsilon, BatchLinUCBDisjointPolicyEpsilon")
  )

  # Compute final estimand
  # B <- dim(expected_rewards)[1]
  B <- nrow(true_linear_rewards)  # Now using subset size (1000)

  estimand <- (1 / B) * sum(policy_probs * true_linear_rewards)

  return(estimand)
}


# BETAS PARAMS OF REWARD MODEL ---------------------------------------------------------------------

get_betas <- function(simulations, d, k) {
  # d: number of features
  # k: number of arms

  list_betas <- lapply(1:(simulations+1), function(i) {
    betas_matrix <- matrix(runif(d * k, -1, 1), d, k)
    betas_matrix <- betas_matrix / norm(betas_matrix, type = "2")
    return(betas_matrix)
  })

  return(list_betas)

}


# CUSTOM CONTEXTUAL LINEAR BANDIT -------------------------------------------------------------------
# store the parameters betas of the observed reward generation model

ContextualLinearBandit <- R6::R6Class(
  "ContextualLinearBandit",
  inherit = Bandit,
  class = FALSE,
  public = list(
    rewards = NULL,
    betas   = NULL,
    sigma   = NULL,
    binary  = NULL,
    weights = NULL,
    list_betas  = NULL,
    sim_id      = NULL,
    class_name = "ContextualLinearBandit",
    initialize  = function(k, d, list_betas, sigma = 0.1, binary_rewards = FALSE) {
      self$k                                    <- k
      self$d                                    <- d
      self$sigma                                <- sigma
      self$binary                               <- binary_rewards
      self$list_betas <- list_betas
    },
    post_initialization = function() {
      # self$betas                                <- matrix(runif(self$d*self$k, -1, 1), self$d, self$k)
      # self$betas                                <- self$betas / norm(self$betas, type = "2")
      # list_betas                                <<- c(list_betas, list(self$betas))
      self$betas <- self$list_betas[[self$sim_id]]

    },
    get_context = function(t) {

      X                                         <- rnorm(self$d)
      self$weights                              <- X %*% self$betas
      reward_vector                             <- self$weights + rnorm(self$k, sd = self$sigma)

      if (isTRUE(self$binary)) {
        self$rewards                            <- rep(0,self$k)
        self$rewards[which_max_tied(reward_vector)] <- 1
      } else {
        self$rewards                            <- reward_vector
      }
      context <- list(
        k = self$k,
        d = self$d,
        X = X
      )
    },
    get_reward = function(t, context_common, action) {
      rewards        <- self$rewards
      optimal_arm    <- which_max_tied(self$weights)
      reward         <- list(
        reward                   = rewards[action$choice],
        optimal_arm              = optimal_arm,
        optimal_reward           = rewards[optimal_arm]
      )
    }
  )
)

# CUSTOM CONTEXTUAL LINEAR POLICIES -----------------------------------------------------------------


# UCB DISJOINT WITH EPSILON
LinUCBDisjointPolicyEpsilon <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    alpha = NULL,
    epsilon = NULL,
    class_name = "LinUCBDisjointPolicyEpsilon",
    initialize = function(alpha = 1.0, epsilon=0.1) {
      super$initialize()
      self$alpha <- alpha
      self$epsilon <- epsilon
    },
    set_parameters = function(context_params) {
      ul <- length(context_params$unique)
      self$theta_to_arms <- list('A' = diag(1,ul,ul), 'b' = rep(0,ul))
    },
    get_action = function(t, context) {

      if (runif(1) > self$epsilon) {

        expected_rewards <- rep(0.0, context$k)

        for (arm in 1:context$k) {

          Xa         <- get_arm_context(context, arm, context$unique)
          A          <- self$theta$A[[arm]]
          b          <- self$theta$b[[arm]]

          A_inv      <- inv(A)

          theta_hat  <- A_inv %*% b

          mu_hat     <- Xa %*% theta_hat
          sigma_hat  <- sqrt(tcrossprod(Xa %*% A_inv, Xa))

          expected_rewards[arm] <- mu_hat + self$alpha * sigma_hat
        }
        action$choice  <- which_max_tied(expected_rewards)

      } else {

        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)
      }

      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa     <- get_arm_context(context, arm, context$unique)

      inc(self$theta$A[[arm]]) <- outer(Xa, Xa)
      inc(self$theta$b[[arm]]) <- reward * Xa

      self$theta
    }
  )
)

# BATCH VERSION OF CONTEXTUAL LINEAR POLICIES ----------------------------------------------------

BatchContextualEpsilonGreedyPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    epsilon = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchContextualEpsilonGreedyPolicy",
    initialize = function(epsilon = 0.1, batch_size=1) {
      super$initialize()
      self$epsilon <- epsilon
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      d <- context_params$d
      k <- context_params$k
      self$theta_to_arms <- list('A' = diag(1,d,d), 'b' = rep(0,d))
      self$A_cc <- replicate(k, diag(1, d, d), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,d), simplify = FALSE)
    },
    get_action = function(t, context) {

      if (runif(1) > self$epsilon) {
        expected_rewards <- rep(0.0, context$k)
        for (arm in 1:context$k) {
          Xa         <- get_arm_context(context, arm)
          A          <- self$theta$A[[arm]]
          b          <- self$theta$b[[arm]]
          A_inv      <- inv(A)
          theta_hat  <- A_inv %*% b
          expected_rewards[arm] <- Xa %*% theta_hat
        }
        action$choice  <- which_max_tied(expected_rewards)
      } else {
        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)
      }

      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa     <- get_arm_context(context, arm)

      self$A_cc[[arm]] <- self$A_cc[[arm]] + outer(Xa, Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)


BatchLinUCBDisjointPolicyEpsilon <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    alpha = NULL,
    epsilon = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchLinUCBDisjointPolicyEpsilon",
    initialize = function(alpha = 1.0, epsilon=0.1, batch_size = 1) {
      super$initialize()
      self$alpha <- alpha
      self$epsilon <- epsilon
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      ul <- length(context_params$unique)
      k <- context_params$k
      self$theta_to_arms <- list('A' = diag(1,ul,ul), 'b' = rep(0,ul))
      self$A_cc <- replicate(k, diag(1, ul, ul), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,ul), simplify = FALSE)
    },
    get_action = function(t, context) {
      if (runif(1) > self$epsilon) {
        expected_rewards <- rep(0.0, context$k)
        for (arm in 1:context$k) {
          Xa         <- get_arm_context(context, arm, context$unique)
          A          <- self$theta$A[[arm]]
          b          <- self$theta$b[[arm]]
          A_inv      <- inv(A)
          theta_hat  <- A_inv %*% b

          mu_hat     <- Xa %*% theta_hat
          sigma_hat  <- sqrt(tcrossprod(Xa %*% A_inv, Xa))

          expected_rewards[arm] <- mu_hat + self$alpha * sigma_hat
        }
        action$choice  <- which_max_tied(expected_rewards)

      } else {
        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)
      }
      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa     <- get_arm_context(context, arm, context$unique)

      self$A_cc[[arm]] <- self$A_cc[[arm]] + outer(Xa, Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)

BatchContextualLinTSPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    sigma = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchContextualLinTSPolicy",
    initialize = function(v = 0.2, batch_size=1) {
      super$initialize()
      self$sigma   <- v^2
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      ul                 <- length(context_params$unique)
      k <- context_params$k
      self$theta_to_arms <- list('A_inv' = diag(1, ul, ul), 'b' = rep(0, ul))
      self$A_cc <- replicate(k, diag(1, ul, ul), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,ul), simplify = FALSE)
    },
    get_action = function(t, context) {
      expected_rewards           <- rep(0.0, context$k)
      for (arm in 1:context$k) {
        Xa                       <- get_arm_context(context, arm, context$unique)
        A_inv                    <- self$theta$A_inv[[arm]]
        b                        <- self$theta$b[[arm]]
        theta_hat                <- A_inv %*% b
        sigma_hat                <- self$sigma * A_inv
        theta_tilde              <- as.vector(contextual::mvrnorm(1, theta_hat, sigma_hat))
        expected_rewards[arm]    <- Xa %*% theta_tilde
      }
      action$choice              <- which_max_tied(expected_rewards)
      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa    <- get_arm_context(context, arm, context$unique)

      self$A_cc[[arm]] <- sherman_morrisson(self$A_cc[[arm]],Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A_inv <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)


#############################################################################################

# FUNCTION UTILITY

#############################################################################################


"inc<-" <- function(x, value) {
  x + value
}

sherman_morrisson <- function(inv, x) {
  inv - c((inv %*% (outer(x, x) %*% inv))) / c(1.0 + (crossprod(x,inv) %*% x))
}

clipr <- function(x, min, max) {
  pmax( min, pmin( x, max))
}

"dec<-" <- function(x, value) {
  x - value
}

which_max_list <- function(x, equal_is_random = TRUE) {
  which_max_tied(unlist(x, FALSE, FALSE), equal_is_random)
}

which_max_tied <- function(x, equal_is_random = TRUE) {
  x <- unlist(x, FALSE, FALSE)
  x <- seq_along(x)[x == max(x)]
  if (length(x) > 1L && equal_is_random)  {
    return(sample(x, 1L, replace = TRUE))
  } else {
    return(x[1])
  }
}

sum_of <- function(x) {
  sum(unlist(x, FALSE, FALSE))
}

inv <- function(M) {
  chol2inv(chol(M))
}

is_rstudio <- function() {
  .Platform$GUI == "RStudio"    #nocov
}


#' @importFrom grDevices graphics.off
#' @importFrom grDevices dev.off
#' @importFrom R.devices devOptions
set_external <- function(ext = TRUE,
                         width = 10,
                         height = 6) {
  # nocov start
  if (is_rstudio()) {
    if (isTRUE(ext)) {
      sysname <- tolower(Sys.info()["sysname"])
      device.name <- "x11"
      switch(sysname,
             darwin = {
               device.name <- "quartz"
             },
             windows = {
               device.name <- "windows"
             })
      options("device" = device.name)
      R.devices::devOptions(sysname, width = width, height = height)
    } else{
      options("device" = "RStudioGD")
    }
    graphics.off()

  }
} # nocov end


sample_one_of <- function(x) {
  if (length(x) <= 1) {
    return(x)
  } else {
    return(sample(x,1))
  }
}


formatted_difftime <- function(x) {
  units(x) <- "secs"
  x <- unclass(x)
  y <- abs(x)
  if (y %/% 86400 > 0) {
    sprintf("%s%d days, %d:%02d:%02d%s",
            ifelse(x < 0, "-", ""), # sign
            y %/% 86400,  # days
            y %% 86400 %/% 3600,  # hours
            y %% 3600 %/% 60,  # minutes
            y %% 60 %/% 1,
            strtrim(substring(as.character(as.numeric(y) %% 1), 2), 4))
  } else {
    sprintf("%s%d:%02d:%02d%s",
            ifelse(x < 0, "-", ""), # sign
            y %% 86400 %/% 3600,  # hours
            y %% 3600 %/% 60,  # minutes
            y %% 60 %/% 1,
            strtrim(substring(as.character(as.numeric(y) %% 1), 2), 4))
  }
}


var_welford <- function(z){
  n = length(z)
  M = list()
  S = list()
  M[[1]] = z[[1]]
  S[[1]] = 0

  for(k in 2:n){
    M[[k]] = M[[k-1]] + ( z[[k]] - M[[k-1]] ) / k
    S[[k]] = S[[k-1]] + ( z[[k]] - M[[k-1]] ) * ( z[[k]] - M[[k]] )
  }
  return(S[[n]] / (n - 1))
}


#' @importFrom stats dgamma
dinvgamma <- function(x, shape, rate = 1, scale = 1/rate, log = FALSE) {
  if(missing(rate) && !missing(scale)) rate <- 1/scale
  log_f <- dgamma(1/x, shape, rate, log = TRUE) - 2*log(x)
  if(log) return(log_f)
  exp(log_f)
}

#' @importFrom stats pgamma
pinvgamma <- function(q, shape, rate = 1, scale = 1/rate, lower.tail = TRUE, log.p = FALSE) {
  if(missing(rate) && !missing(scale)) rate <- 1/scale
  pgamma(1/q, shape, rate, lower.tail = !lower.tail, log.p = log.p)
}

#' @importFrom stats qgamma
qinvgamma <- function(p, shape, rate = 1, scale = 1/rate, lower.tail = TRUE, log.p = FALSE) {
  if(missing(rate) && !missing(scale)) rate <- 1/scale
  qgamma(1-p, shape, rate, lower.tail = lower.tail, log.p = log.p)^(-1)
}

#' @importFrom stats rgamma
rinvgamma <- function(n, shape, rate = 1, scale = 1/rate) {
  if(missing(rate) && !missing(scale)) rate <- 1/scale
  1 / rgamma(n, shape, rate)
}


invlogit <- function(x){
  exp(x)/(1+exp(x))
}


ones_in_zeroes <- function(vector_length, index_of_one) {
  x <- rep(0, vector_length)
  x[index_of_one] <- 1
  return(x[1:vector_length])
}

get_arm_context <- function(context, arm, select_features = NULL, prepend_arm_vector = FALSE) {
  # X <- as.numeric(levels(X))[X]
  X <- context$X
  k <- context$k
  if(is.null(select_features)) {
    if(is.vector(X)) Xv <- X else Xv <- X[, arm]
  } else {
    if(is.vector(X)) Xv <- X[select_features]
    else Xv <- X[select_features, arm]
  }
  if(isTRUE(prepend_arm_vector)) Xv <- c(ones_in_zeroes(k,arm),Xv)
  return(Xv)
}


get_full_context <- function(context, select_features = NULL, prepend_arm_matrix = FALSE) {
  X <- context$X
  d <- context$d
  k <- context$k
  if(is.null(select_features)) {
    if(is.vector(X)) Xm <- matrix(X,d,k) else Xm <- X
  } else {
    if(is.vector(X)) Xm <- X[select_features]
    else Xm <- X[select_features,]
  }
  if(isTRUE(prepend_arm_matrix)) Xv <- rbind(diag(k),Xv)
  return(Xm)
}


one_hot <- function(dt, cols="auto", sparsifyNAs=FALSE, naCols=FALSE, dropCols=TRUE, dropUnusedLevels=FALSE){
  # One-Hot-Encode unordered factors in a data.table
  # If cols = "auto", each unordered factor column in dt will be encoded.
  # (Or specifically a vector of column names to encode)
  # If dropCols=TRUE, the original factor columns are dropped
  # If dropUnusedLevels = TRUE, unused factor levels are dropped

  #--------------------------------------------------
  # Hack to pass 'no visible binding for global variable' notes from R CMD check

  OHEID <- NULL

  #--------------------------------------------------

  # Automatically get the unordered factor columns
  if(cols[1] == "auto") cols <- colnames(dt)[which(sapply(dt, function(x) is.factor(x) & !is.ordered(x)))]

  # If there are no columns to encode, return dt
  if(length(cols) == 0) return(dt)

  # Build tempDT containing and ID column and 'cols' columns
  tempDT <- dt[, cols, with=FALSE]
  tempDT[, OHEID := .I]
  for(col in cols) set(tempDT, j=col, value=factor(paste(col, tempDT[[col]], sep="_"),
                                                   levels=paste(col, levels(tempDT[[col]]), sep="_")))

  # One-hot-encode
  melted <- melt(tempDT, id = 'OHEID', value.factor = T, na.rm=TRUE)
  if(dropUnusedLevels == TRUE){
    newCols <- dcast(melted, OHEID ~ value, drop = T, fun.aggregate = length)
  } else{
    newCols <- dcast(melted, OHEID ~ value, drop = F, fun.aggregate = length)
  }

  # Fill in potentially missing rows
  newCols <- newCols[tempDT[, list(OHEID)]]
  newCols[is.na(newCols[[2]]), setdiff(paste(colnames(newCols)), "OHEID") := 0L]

  #--------------------------------------------------
  # Deal with NAs

  if(!sparsifyNAs | naCols){

    # Determine which columns have NAs
    na_cols <- character(0)
    for(col in cols) if(any(is.na(tempDT[[col]]))) na_cols <- c(na_cols, col)

    # If sparsifyNAs is TRUE, find location of NAs in dt and insert them in newCols
    if(!sparsifyNAs)
      for(col in na_cols) newCols[is.na(tempDT[[col]]), intersect(levels(tempDT[[col]]),
                                                                  colnames(newCols)) := NA_integer_]

    # If naCols is TRUE, build a vector for each column with an NA value and 1s indicating the location of NAs
    if(naCols)
      for(col in na_cols) newCols[, eval(paste0(col, "_NA")) := is.na(tempDT[[col]]) * 1L]
  }

  #--------------------------------------------------
  # Clean Up

  # Combine binarized columns with the original dataset
  result <- cbind(dt, newCols[, !"OHEID"])

  # Reorder columns
  possible_colnames <- character(0)
  for(col in colnames(dt)){
    possible_colnames <- c(possible_colnames, col)
    if(col %in% cols){
      possible_colnames <- c(possible_colnames, paste0(col, "_NA"))
      possible_colnames <- c(possible_colnames, paste(levels(tempDT[[col]])))
    }
  }
  sorted_colnames <- intersect(possible_colnames, colnames(result))
  setcolorder(result, sorted_colnames)

  # If dropCols = TRUE, remove the original factor columns
  if(dropCols == TRUE) result <- result[, !cols, with=FALSE]

  return(result)
}


mvrnorm = function(n, mu, sigma)
{
  ncols <- ncol(sigma)
  mu <- rep(mu, each = n)
  mu + matrix(stats::rnorm(n * ncols), ncol = ncols) %*% chol(sigma)
}


value_remaining <- function(x, n, alpha = 1, beta = 1, ndraws = 10000)
{
  post <- sim_post(x,n,alpha,beta,ndraws)
  postWin <- prob_winner(post)
  iMax <- which.max(postWin)
  thetaMax <- apply(post,1,max)
  #value_remaining:
  vR <- (thetaMax-post[,iMax])/post[,iMax]
  return(vR)
}


sim_post <- function(x, n, alpha = 1, beta = 1, ndraws = 5000) {
  k <- length(x)
  ans <- matrix(nrow=ndraws, ncol=k)
  no <- n-x
  for (i in (1:k))
    ans[,i] <- stats::rbeta(ndraws, x[i] + alpha, no[i] + beta)
  return(ans)
}


prob_winner <- function(post){
  k <- ncol(post)
  w <- table(factor(max.col(post), levels = 1:k))
  return(w/sum(w))
}


ind <- function(cond) {
  ifelse(cond, 1L, 0L)
}


data_table_factors_to_numeric <- function(dt){
  setDT(dt)
  factor_cols <- names(which(sapply(dt, class)=="factor"))
  if(length(factor_cols) > 0) {
    suppressWarnings(dt[,(factor_cols) :=
                          lapply(.SD, function(x) as.numeric(as.character(x))),.SDcols=factor_cols])
  }
  return(dt)
}


get_global_seed = function() {
  current.seed = NA
  if (exists(".Random.seed", envir=.GlobalEnv)) {
    current.seed = .Random.seed
  }
  current.seed
}


set_global_seed = function(x) {
  if (length(x)>1) {
    assign(".Random.seed", x, envir=.GlobalEnv)
  }
}



#############################################################################################

# FUNCTION GENERIC

#############################################################################################

#' @export
plot.History <- function(x, ...) {
  args <- eval(substitute(alist(...)))
  if ("type" %in% names(args)) {
    type <- eval(args$type)
  } else {
    type <- "cumulative"
  }
  if ("xlim" %in% names(args))
    xlim <- eval(args$xlim)
  else
    xlim <- NULL
  if ("legend" %in% names(args))
    legend <- eval(args$legend)
  else
    legend <- TRUE
  if ("trunc_per_agent" %in% names(args))
    trunc_per_agent <- eval(args$trunc_per_agent)
  else
    trunc_per_agent <- TRUE
  if ("trunc_over_agents" %in% names(args))
    trunc_over_agents <- eval(args$trunc_over_agents)
  else
    trunc_over_agents <- TRUE
  if ("regret" %in% names(args))
    regret <- eval(args$regret)
  else
    regret <- TRUE
  if ("use_colors" %in% names(args))
    use_colors <- eval(args$use_colors)
  else
    use_colors <- TRUE
  if ("log" %in% names(args))
    log <- eval(args$log)
  else
    log <- ""
  if ("plot_only_disp" %in% names(args))
    plot_only_disp <- eval(args$plot_only_disp)
  else
    plot_only_disp <- FALSE
  if ("disp" %in% names(args))
    disp <- eval(args$disp)
  else
    disp <- NULL
  if ("traces" %in% names(args))
    traces <- eval(args$traces)
  else
    traces <- FALSE
  if ("traces_alpha" %in% names(args))
    traces_alpha <- eval(args$traces_alpha)
  else
    traces_alpha <- 0.3
  if ("traces_max" %in% names(args))
    traces_max <- eval(args$traces_max)
  else
    traces_max <- 100
  if ("smooth" %in% names(args))
    smooth <- eval(args$smooth)
  else
    smooth <- FALSE
  if ("interval" %in% names(args))
    interval <- eval(args$interval)
  else
    interval <- 1
  if ("color_step" %in% names(args))
    color_step <- eval(args$color_step)
  else
    color_step <- 1
  if ("lty_step" %in% names(args))
    lty_step <- eval(args$lty_step)
  else
    lty_step <- 1
  if ("lwd" %in% names(args))
    lwd <- eval(args$lwd)
  else
    lwd <- 2
  if ("ylim" %in% names(args))
    ylim <- eval(args$ylim)
  else
    ylim <- NULL
  if ("legend_labels" %in% names(args))
    legend_labels <- eval(args$legend_labels)
  else
    legend_labels <- NULL
  if ("legend_position" %in% names(args))
    legend_position <- args$legend_position
  else
    if (type == "arms")
      legend_position <- "bottomright"
    else
      legend_position <- "topleft"
    if ("limit_agents" %in% names(args))
      limit_agents <- eval(args$limit_agents)
    else
      limit_agents <- NULL
    if ("limit_context" %in% names(args))
      limit_context <- eval(args$limit_context)
    else
      limit_context <- NULL
    if ("legend_border" %in% names(args))
      legend_border <- eval(args$legend_border)
    else
      legend_border <- NULL
    if ("cum_average" %in% names(args))
      cum_average <- eval(args$cum_average)
    else
      cum_average <- FALSE
    if ("legend_title" %in% names(args))
      legend_title <- eval(args$legend_title)
    else
      legend_title <- NULL
    if ("xlab" %in% names(args))
      xlab <- eval(args$xlab)
    else
      xlab <- NULL
    if ("ylab" %in% names(args))
      ylab <- eval(args$ylab)
    else
      ylab <- NULL
    if ("rate" %in% names(args))
      rate <- eval(args$rate)
    else
      rate <- FALSE
    if ("no_par" %in% names(args))
      no_par <- eval(args$no_par)
    else
      no_par <- FALSE
    ### checkmate::assert_choice(type, c("cumulative","average","arms")) TODO: fix checkmate
    if (type == "cumulative") {
      Plot$new()$cumulative(
        x,
        xlim = xlim,
        legend = legend,
        regret = regret,
        use_colors = use_colors,
        log = log,
        disp = disp,
        plot_only_disp = plot_only_disp,
        traces = traces,
        traces_max = traces_max,
        traces_alpha = traces_alpha,
        smooth = smooth,
        interval = interval,
        color_step = color_step,
        lty_step = lty_step,
        lwd = lwd,
        rate = rate,
        ylim = ylim,
        legend_labels = legend_labels,
        legend_border = legend_border,
        legend_position = legend_position,
        legend_title = legend_title,
        no_par = no_par,
        xlab = xlab,
        ylab = ylab,
        limit_agents = limit_agents,
        limit_context = limit_context,
        trunc_over_agents = trunc_over_agents,
        trunc_per_agent = trunc_per_agent
      )
    } else if (type == "average") {
      Plot$new()$average(
        x,
        xlim = xlim,
        legend = legend,
        regret = regret,
        log = log,
        disp = disp,
        plot_only_disp = plot_only_disp,
        traces = traces,
        traces_max = traces_max,
        traces_alpha = traces_alpha,
        smooth = smooth,
        interval = interval,
        color_step = color_step,
        lty_step = lty_step,
        lwd = lwd,
        rate = rate,
        ylim = ylim,
        legend_labels = legend_labels,
        legend_border = legend_border,
        legend_position = legend_position,
        legend_title = legend_title,
        no_par = no_par,
        xlab = xlab,
        ylab = ylab,
        cum_average = cum_average,
        limit_agents = limit_agents,
        limit_context = limit_context,
        trunc_over_agents = trunc_over_agents,
        trunc_per_agent = trunc_per_agent
      )
    } else if (type == "optimal") {
      Plot$new()$optimal(
        x,
        xlim = xlim,
        legend = legend,
        log = log,
        disp = disp,
        plot_only_disp = plot_only_disp,
        traces = traces,
        traces_max = traces_max,
        traces_alpha = traces_alpha,
        smooth = smooth,
        interval = interval,
        color_step = color_step,
        lty_step = lty_step,
        lwd = lwd,
        ylim = ylim,
        legend_labels = legend_labels,
        legend_border = legend_border,
        legend_position = legend_position,
        legend_title = legend_title,
        no_par = no_par,
        xlab = xlab,
        ylab = ylab,
        limit_agents = limit_agents,
        limit_context = limit_context,
        trunc_over_agents = trunc_over_agents,
        trunc_per_agent = trunc_per_agent
      )
    } else if (type == "arms") {
      Plot$new()$arms(
        x,
        xlim = xlim,
        legend = legend,
        use_colors = use_colors,
        log = log,
        interval = interval,
        ylim = ylim,
        smooth = smooth,
        legend_labels = legend_labels,
        legend_border = legend_border,
        legend_position = legend_position,
        legend_title = legend_title,
        no_par = no_par,
        xlab = xlab,
        ylab = ylab,
        trunc_over_agents = trunc_over_agents,
        limit_agents = limit_agents,
        limit_context = limit_context
      )
    }
}

#' @export
print.History <- function(x, ...) {
  summary.History(x)
}

#' @export
summary.History <- function(object, ...) {

  args <- eval(substitute(alist(...)))
  if ("limit_agents" %in% names(args))
    limit_agents <- eval(args$limit_agents)
  else
    limit_agents <- NULL

  cum <- object$get_cumulative_result(limit_agents=limit_agents, as_list = FALSE)
  cum$sims <- object$get_simulation_count()

  cat("\nAgents:\n\n")
  agents <- object$get_agent_list()
  cat(paste(' ', agents, collapse = ', '))

  cat("\n\nCumulative regret:\n\n")
  print(cum[,c("agent","t", "sims", "cum_regret", "cum_regret_var",
               "cum_regret_sd")], fill = TRUE, row.names = FALSE)

  cat("\n\nCumulative reward:\n\n")
  print(cum[,c("agent","t", "sims", "cum_reward", "cum_reward_var",
               "cum_reward_sd")], fill = TRUE, row.names = FALSE)

  cat("\n\nCumulative reward rate:\n\n")
  crr <- cum[,c("agent","t", "sims", "cum_reward_rate", "cum_reward_rate_var",
                "cum_reward_rate_sd")]
  names(crr) <- c("agent","t", "sims", "cur_reward", "cur_reward_var",
                  "cur_reward_sd")
  print(crr, fill = TRUE, row.names = FALSE)


  cat("\n")
}
