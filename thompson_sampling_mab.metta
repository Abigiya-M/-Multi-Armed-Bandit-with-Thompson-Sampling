;; ============================================================
;; Multi-Armed Bandit with Thompson Sampling in MeTTa
;; ============================================================
;;
;; PROBLEM: Online Ad Selection
;; ----------------------------
;; A website shows one of 4 ads per visit. Each ad has an
;; unknown click-through rate (CTR). The goal is to maximise
;; total clicks by learning which ad performs best.
;;
;;   Arm A  (true CTR ≈ 0.70)  ← best arm
;;   Arm B  (true CTR ≈ 0.50)
;;   Arm C  (true CTR ≈ 0.30)
;;   Arm D  (true CTR ≈ 0.10)
;;
;; METHOD: Thompson Sampling (Beta-Bernoulli)
;;   Prior per arm:  Beta(1, 1) — uniform
;;   Each round:
;;     1. Sample a score from each arm's posterior
;;     2. Pick the arm with the highest score
;;     3. Observe binary reward (click / no click)
;;     4. Update: success → α+1, failure → β+1
;;
;; We approximate Beta sampling by:
;;   score = (α / (α+β)) × 1000 + uniform_noise(−50,+50)
;; ============================================================

;; ---------- random via Python --------------------------------
(= (randint $lo $hi) ((py-atom random.randint) $lo $hi))

;; ---------- arm posteriors -----------------------------------
!(bind! &arms (new-space))
!(add-atom &arms (arm A 1 1))
!(add-atom &arms (arm B 1 1))
!(add-atom &arms (arm C 1 1))
!(add-atom &arms (arm D 1 1))

;; ---------- environment (hidden CTRs) ------------------------
(= (pull A) (if (<= (randint 1 100) 70) 1 0))
(= (pull B) (if (<= (randint 1 100) 50) 1 0))
(= (pull C) (if (<= (randint 1 100) 30) 1 0))
(= (pull D) (if (<= (randint 1 100) 10) 1 0))

;; ---------- Thompson score -----------------------------------
;; Wider noise (±150) forces more exploration early on,
;; then as α+β grows the mean dominates and exploitation wins.
(= (score $id)
    (let* (
        (($a $b) (match &arms (arm $id $a $b) ($a $b)))
    ) (+ (/ (* $a 1000) (+ $a $b)) (- (randint 0 300) 150))))

;; ---------- arm selection ------------------------------------
(= (bestArm)
    (let* (
        ($s1 (score A))
        ($s2 (score B))
        ($s3 (score C))
        ($s4 (score D))
    ) (if (and (>= $s1 $s2) (and (>= $s1 $s3) (>= $s1 $s4))) A
      (if (and (>= $s2 $s3) (>= $s2 $s4)) B
      (if (>= $s3 $s4) C D)))))

;; ---------- posterior update ---------------------------------
(= (update $id $reward)
    (let* (
        (($a $b) (match &arms (arm $id $a $b) ($a $b)))
        ($na (if (== $reward 1) (+ $a 1) $a))
        ($nb (if (== $reward 0) (+ $b 1) $b))
        ($_ (remove-atom &arms (arm $id $a $b)))
        ($_ (add-atom &arms (arm $id $na $nb)))
    ) ()))

;; ---------- main loop ----------------------------------------
(= (run $n)
    (if (== $n 0) ()
    (let* (
        ($chosen (bestArm))
        ($reward (pull $chosen))
        ($_ (update $chosen $reward))
        ($_ (println! ("Round" $n $chosen $reward)))
    ) (run (- $n 1)))))

;; ---------- run experiment -----------------------------------
!(println! "=== Thompson Sampling - Online Ad Selection ===")
!(run 30)
!(println! "=== Final posteriors (arm alpha beta) ===")
!(match &arms (arm $id $a $b) ($id $a $b))
