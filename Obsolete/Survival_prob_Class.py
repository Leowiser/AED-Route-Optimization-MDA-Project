class survival_prob:
    # this is based on the dict obtained from the output in fastest_comparisson in Simulation_Class 

        # only used to compare, so we do not need to check if vector or responder arrives first, just comapre their prop of survival
        # this is the probability of survival immedaiatly after cpr or aed chock, not the survival after 24 hours... 
        # info on survival rates from: https://www.sciencedirect.com/science/article/pii/S0019483219304080
        # and from here: https://jamanetwork.com/journals/jama/fullarticle/196200
        # and from here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2600120/
    def probability_survival(self, duration_responder, duration_AED, duration_vector, decrease_with_cpr = 0.97, decrease_no_cpr = 0.9):
        # explanation of parameters 
        # duration_responder: time it takes for responder without aed to arrive, integer
        # duration_AED: for aed to arrive, integer
        # duration_vector: for vector to arrive, integer
        # decrease_with_cpr: decrease in survival if cpr is perfromed  
        # decrease_no_cpr: decrease in survival if cpr is not performed 

        prob_resp = 1 # starting probability to survive with responder
        prob_vec = 1 # starting probability to survive with vector 

        # calcualte probability of survival with vector, probability decreases with decrease_no_cpr per minute 
        time_vec_min = duration_vector/60 
        prob_vec = prob_vec * decrease_no_cpr ** time_vec_min

        # also add vector to calcualtion since we never have only first responder 
        fastest_aed = min(duration_AED, duration_vector) # get the fastest of the aed and vector that can arrive with aed 

        if(fastest_aed <= duration_responder): # if AED/vector is faster or as fast as first responder, no cpr is started
            time_aed_min = fastest_aed/60 # time to aed arrives in min 
            prob_resp = prob_resp * decrease_no_cpr ** time_aed_min
        else: # if responder is faster than aed 
            # time without cpr in min
            time_no_cpr = duration_responder/60
            # time with cpr before aed arrives in min 
            time_with_cpr = (fastest_aed-duration_responder)/60
            # decrease in survival when cpr is started after a while   
            prob_resp = prob_resp * decrease_no_cpr ** time_no_cpr * decrease_with_cpr**time_with_cpr
        return prob_resp, prob_vec 
