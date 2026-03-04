;; ============================================================
;; Multi-Armed Bandit with Thompson Sampling (Safe PeTTa)
;; Proper numeric termination
;; ============================================================

;; ---------- Pseudo-random number generation -------------------
(= (randint $lo $hi)
    (+ $lo (% (+ (* $lo 7) 9817) (- $hi $lo))))

;; ---------- Environment ----------------------------------------
(= (pull-A) (if (<= (randint 1 100) 70) 1 0))
(= (pull-B) (if (<= (randint 1 100) 50) 1 0))
(= (pull-C) (if (<= (randint 1 100) 30) 1 0))
(= (pull-D) (if (<= (randint 1 100) 10) 1 0))

(= (pull A) (pull-A))
(= (pull B) (pull-B))
(= (pull C) (pull-C))
(= (pull D) (pull-D))

;; ---------- Initial state --------------------------------------
(= (init) ((A 1 1) (B 1 1) (C 1 1) (D 1 1)))

;; ---------- Get parameters -------------------------------------
(= (get A ((A $a $b) $_ $_ $_)) ($a $b))
(= (get B (($_ $_ $_) (B $a $b) $_ $_)) ($a $b))
(= (get C (($_ $_ $_) ($_ $_ $_) (C $a $b) $_)) ($a $b))
(= (get D (($_ $_ $_) ($_ $_ $_) ($_ $_ $_) (D $a $b))) ($a $b))

;; ---------- Thompson score -------------------------------------
(= (score $id $state)
    (let* (
        (($a $b) (get $id $state))
        ($mean (/ (* $a 1000) (+ $a $b)))
        ($noise (- (randint 0 300) 150))
    )
        (+ $mean $noise)))

;; ---------- Choose best arm ------------------------------------
(= (choose-arm $state)
    (let* (
        ($sa (score A $state))
        ($sb (score B $state))
        ($sc (score C $state))
        ($sd (score D $state))
    )
        (if (>= $sa $sb)
            (if (>= $sa $sc)
                (if (>= $sa $sd) A (if (>= $sc $sd) C D))
                (if (>= $sc $sd) C D))
            (if (>= $sb $sc)
                (if (>= $sb $sd) B (if (>= $sc $sd) C D))
                (if (>= $sc $sd) C D)))))

;; ---------- Update state ---------------------------------------
(= (upd A 1 ((A $a $b) $s2 $s3 $s4))
    ((A (+ $a 1) $b) $s2 $s3 $s4))
(= (upd A 0 ((A $a $b) $s2 $s3 $s4))
    ((A $a (+ $b 1)) $s2 $s3 $s4))

(= (upd B 1 ($s1 (B $a $b) $s3 $s4))
    ($s1 (B (+ $a 1) $b) $s3 $s4))
(= (upd B 0 ($s1 (B $a $b) $s3 $s4))
    ($s1 (B $a (+ $b 1)) $s3 $s4))

(= (upd C 1 ($s1 $s2 (C $a $b) $s4))
    ($s1 $s2 (C (+ $a 1) $b) $s4))
(= (upd C 0 ($s1 $s2 (C $a $b) $s4))
    ($s1 $s2 (C $a (+ $b 1)) $s4))

(= (upd D 1 ($s1 $s2 $s3 (D $a $b)))
    ($s1 $s2 $s3 (D (+ $a 1) $b)))
(= (upd D 0 ($s1 $s2 $s3 (D $a $b)))
    ($s1 $s2 $s3 (D $a (+ $b 1))))

;; ---------- Safe Simulation Loop --------------------------------
(= (sim-loop $round $state)
    (if (> $round 30)
        (println! (list "Final state:" $state))
        (let* (
            ($chosen (choose-arm $state))
            ($reward (pull $chosen))
            ($new-state (upd $chosen $reward $state))
            ($_ (println! (list $round $chosen $reward)))
            ($next (+ $round 1))
        )
            (sim-loop $next $new-state))))

;; ---------- Run Experiment --------------------------------------
!(println! "=== Thompson Sampling (Safe Version) ===")
!(sim-loop 1 (init))