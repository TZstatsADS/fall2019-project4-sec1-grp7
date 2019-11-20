#### a function returns RMSE
RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[2]), as.character(obs[1])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

#### pmf function returns a list containing factorized matrices p and q, training and testing RMSEs.
PMF <- function(f = 10, 
                lrate = 0.001, max.iter = 100,
                data, train, test, stopping.deriv = 0.1,
                lambdap = 0.1, lambdaq = 0.2) {
  set.seed(123)
  #random assign value to matrix p and q
  p <- matrix(runif(f*U, -1, 1), ncol = U) 
  grad_p <- matrix(0, ncol = U, nrow = f)
  colnames(p) <- colnames(grad_p) <-  levels(as.factor(data$userId))
  q <- matrix(runif(f*I, -1, 1), ncol = I)
  grad_q <- matrix(0, ncol = I, nrow = f) 
  colnames(q) <- colnames(grad_q) <- levels(as.factor(data$movieId))
  
  RMSE_df <- matrix(ncol = 2, nrow = 0)
  
  for (l in 1:max.iter){
    #loop through each sample when I{ui = 1} 
    for(i in 1:I){
      grad_q[, i] <- - lambdaq * q[, i]
    }
    for(u in 1:U){
      grad_p[, u] <- - lambdap * p[, u]
    }
    for (s in 1:nrow(train)){
      u <- as.character(train[s,"userId"])
      i <- as.character(train[s,"movieId"])
      e_ui <- train[s,"rating"] - t(q[, i]) %*% p[, u]
      grad_q[, i] <- grad_q[, i] + e_ui %*% p[, u] 
      grad_p[, u] <- grad_p[, u] + e_ui %*% q[, i]
    }
    
    if(l == 1){
      f_norm = sqrt(sum(grad_p^2) + sum(grad_q^2))
    }
    
    q = q + lrate * grad_q
    p = p + lrate * grad_p
    
    if (sqrt(sum(grad_p^2) + sum(grad_q^2)) < stopping.deriv * f_norm){
      break
    }
    
    print(sqrt(sum(grad_p^2) + sum(grad_q^2)))
    
    cat("epoch:", l, "\t")
    est_rating <- t(q) %*% p
    rownames(est_rating) <- levels(as.factor(data$movieId))
    
    RMSE_df <- rbind(RMSE_df, c(RMSE(train, est_rating), RMSE(test, est_rating)))
    cat("training RMSE:", RMSE_df[l, 1], "\t")
    
    cat("test RMSE:", RMSE_df[l, 2], "\n")
    
  }
  
  return(list(p = p, q = q, train_RMSE = RMSE_df[, 1], test_RMSE = RMSE_df[, 2]))
}


update_PMF <- function(f = 10, 
                       lrate = 0.001, max.iter = 100,
                       data, train, test, stopping.deriv = 0.1,
                       lambdap = 0.1, lambdaq = 0.2) {
  set.seed(123)
  #random assign value to matrix p and q
  p <- matrix(runif(f*U, -1, 1), ncol = U) 
  grad_p <- matrix(0, ncol = U, nrow = f)
  colnames(p) <- colnames(grad_p) <-  levels(as.factor(data$userId))
  q <- matrix(runif(f*I, -1, 1), ncol = I)
  grad_q <- matrix(0, ncol = I, nrow = f) 
  colnames(q) <- colnames(grad_q) <- levels(as.factor(data$movieId))
  list_p <- list_q <- rep(list(diag(0)), max.iter)
  RMSE_df <- matrix(ncol = 2, nrow = 0)
  
  for (l in 1:max.iter){
    #loop through each sample when I{ui = 1} 
    for(i in 1:I){
      grad_q[, i] <- - lambdaq * q[, i]
    }
    for(u in 1:U){
      grad_p[, u] <- - lambdap * p[, u]
    }
    for (s in 1:nrow(train)){
      u <- as.character(train[s,"userId"])
      i <- as.character(train[s,"movieId"])
      e_ui <- train[s,"rating"] - t(q[, i]) %*% p[, u]
      grad_q[, i] <- grad_q[, i] + e_ui %*% p[, u] 
      grad_p[, u] <- grad_p[, u] + e_ui %*% q[, i]
    }
    
    if(l == 1){
      f_norm = sqrt(sum(grad_p^2) + sum(grad_q^2))
    }
    
    q = q + lrate * grad_q
    p = p + lrate * grad_p
    
    list_p[[l]] <- p
    list_q[[l]] <- q
    
    if (sqrt(sum(grad_p^2) + sum(grad_q^2)) < stopping.deriv * f_norm){
      break
    }
    
    print(sqrt(sum(grad_p^2) + sum(grad_q^2)))
    
    cat("epoch:", l, "\t")
    est_rating <- t(q) %*% p
    rownames(est_rating) <- levels(as.factor(data$movieId))
    
    RMSE_df <- rbind(RMSE_df, c(RMSE(train, est_rating), RMSE(test, est_rating)))
    cat("training RMSE:", RMSE_df[l, 1], "\t")
    
    cat("test RMSE:", RMSE_df[l, 2], "\n")
    
  }
  
  return(list(p = list_p, q = list_q, train_RMSE = RMSE_df[, 1], test_RMSE = RMSE_df[, 2]))
}


#### als function returns a list containing factorized matrices p and q, training and testing RMSEs.
als <- function(feature= 2, lambda = 0.1, max.iter = 1, stopping.deriv = 0.1, data, train, test){
  set.seed(0)
  #random assign value to matrix (user)p and (movie)q
  p <- matrix(runif(feature*U, -1, 1), ncol = U) 
  colnames(p) <- as.character(1:U)
  q <- matrix(runif(feature*I, -1, 1), ncol = I)
  colnames(q) <- as.character(sort(unique(data$movieId),decreasing = FALSE))
  for (t in 1:I){
    if(!is.na(ratingmean[t])) {q[1,t]=ratingmean[t]}
    else{q[1,t]=mean(data_train$rating)}
  }
  
  train_RMSE <- c()
  test_RMSE <- c()
  
  for(l in 1:max.iter){
    # Fix q,get p
    for (u in 1:U){
      u_row <- train[train$userId==u,]
      Mu<-q[,as.character(u_row$movieId)]
      Au<-Mu%*%t(Mu)+lambda*dim(u_row)[1]*diag(feature)
      
      Ru<-u_row$rating
      Vu<-Mu%*%Ru
      p[,u]<-solve(Au)%*%Vu
    }
    
    # Fix p,get q
    for (i in 1:I){
      i_row <- train[train$movieId==colnames(q)[i],]
      Mi<-p[,as.character(i_row$userId)]
      Ai<-Mi%*%t(Mi)+lambda*dim(i_row)[1]*diag(feature)
      
      Ri<-i_row$rating
      if (length(Ri)==1){
        Vi<-Mi*Ri
        q[,i]<-solve(Ai)%*%Vi
      }
      if (length(Ri)>1){
        Vi<-Mi%*%Ri
        q[,i]<-solve(Ai)%*%Vi
      }
    }
    
    #get RMSE
    
    cat("epoch:", l, "\t")
    est_rating <- t(q) %*% p
    rownames(est_rating) <- levels(as.factor(data$movieId))
    
    train_RMSE_cur <- RMSE(train, est_rating)
    cat("training RMSE:", train_RMSE_cur, "\t")
    train_RMSE <- c(train_RMSE, train_RMSE_cur)
    
    test_RMSE_cur <- RMSE(test, est_rating)
    cat("test RMSE:",test_RMSE_cur, "\n")
    test_RMSE <- c(test_RMSE, test_RMSE_cur)
    
  }
  return(list(p = p, q = q, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
}


#### pmf cv function
pmf_cv.function <- function(data, dat_train, K, f, lq, lp, lrate = 0.001, max.iter = 20){
  ### Input:
  ### - train data frame
  ### - K: a number stands for K-fold CV
  ### - tuning parameters 
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  df_rmse <- matrix(0.1, ncol = 2, nrow = K)
  #test_rmse <- matrix(NA, ncol = 10, nrow = K)
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- PMF(f = f, lambdap = lp, lambdaq = lq,
                  lrate = lrate, max.iter = max.iter, stopping.deriv = 0.1,
                  data = data, train = train.data, test = test.data)
    
    df_rmse[i,1] <-  result$train_RMSE[which(result$test_RMSE == min(result$test_RMSE))]
    df_rmse[i,2] <- min(result$test_RMSE)
    #df_rmse[i,] <-  c(f, lq, lp, result$train_RMSE[which(result$test_RMSE == min(result$test_RMSE))], min(result$test_RMSE))
    result <- c(f, lq, lp, mean(df_rmse[,1]), mean(df_rmse[,2]))
    names(result) <- c("f", "lq", 'lp', "train_rmse", "test_rmse")
  }		
  return(result)
}


#### als cv function
als_cv.function <- function(dat_train, K, feature, lambda){
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  train_rmse <- c()
  test_rmse <- c()
  
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- als( lambda = lambda, feature = feature,
                   stopping.deriv = 0.01, data = data, train = train.data, test = test.data)
    
    train_rmse[i] <-  result$train_rmse
    test_rmse[i] <-   result$test_rmse
    
  }		
  return( list(mean_train_rmse = mean(train_rmse), mean_test_rmse = mean(test_rmse),
               sd_train_rmse = sd(train_rmse), sd_test_rmse = sd(test_rmse)) )
}


#### post processing function returns kernel ridge regression predictions
norm <- function(x) {x / sqrt(sum(x^2))}
y_hat_kernel<-function(movie_vec, kernel, lambda, f, data_train, data_test){
  
  Y_train_kernel<-NA
  Y_test_kernel<-NA
  
  for (i in 1:range(data_test$userId)[2]) {
    
    ##Select movie features for user i,get Xtrain train matrix
    movie_ID_1<-data_train[data_train$userId==i,]$movieId
    oo<-movie_vec[,as.character(movie_ID_1)]
    if(length(oo)>f){X_train<-t(apply(oo, 2, norm))}
    if(length(oo)==f){X_train<-t(norm(oo))}
    ##Get y(rating) vector
    y<-matrix(data_train[data_train$userId==i,]$rating,ncol = 1)
    
    
    ##Select movie features for user i,get Xtest train matrix
    movie_ID_2<-data_test[data_test$userId==i,]$movieId
    oo<-movie_vec[,as.character(movie_ID_2)]
    if(length(oo)>f){X_test<-t(apply(oo, 2, norm))}
    if(length(oo)==f){X_test<-t(norm(oo))}
    
    ##Do prediction
    yi_train_kernel<-kernelMatrix(kernel, X_train, X_train)%*%solve(kernelMatrix(kernel, X_train, X_train)+lambda*diag(dim(X_train)[1]))%*%y
    yi_test_kernel<-kernelMatrix(kernel, X_test, X_train)%*%solve(kernelMatrix(kernel, X_train, X_train)+lambda*diag(dim(X_train)[1]))%*%y
    
    
    ##Combine predictions
    train_kernel<-matrix(c(rep(i,length(yi_train_kernel)),movie_ID_1,yi_train_kernel),nrow = length(yi_train_kernel),ncol = 3)
    Y_train_kernel<-rbind(Y_train_kernel,train_kernel)
    
    test_kernel<-matrix(c(rep(i,length(yi_test_kernel)),movie_ID_2,yi_test_kernel),nrow = length(yi_test_kernel),ncol = 3)
    Y_test_kernel<-rbind(Y_test_kernel,test_kernel)
    
  }
  colnames(Y_train_kernel)<-c("userID","movieID","rating")
  colnames(Y_test_kernel)<-c("userID","movieID","rating")
  return(list(Y_train_kernel[-1,],Y_test_kernel[-1,]))
}