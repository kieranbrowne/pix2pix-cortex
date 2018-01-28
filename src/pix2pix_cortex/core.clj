(ns pix2pix-cortex.core
  (:require [cortex.experiment.train :as train]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [clisk.live :as clisk]
            [mikera.image.core :as img]))

(defn mandelbrot [x y r]
  (clisk/image
   (clisk/viewport
    [(- x r) (- y r)] [(+ x r) (+ y r)]
    (clisk/fractal
     :while (clisk/v- 2 (clisk/length [clisk/x clisk/y]))
     :update (clisk/v+ clisk/c [(clisk/v- (clisk/v* clisk/x clisk/x) (clisk/v* clisk/y clisk/y)) (clisk/v* 2 clisk/x clisk/y)])
     :result (clisk/vplasma (clisk/v* 0.1 'i))
     :bailout-result clisk/black
     :max-iterations 1000
     )) :width 400 :height 400))


(img/show
 (mandelbrot -1.31 0 00.1))

(img/show
 (mandelbrot -0.74603 0.11002 0.00000001001))

(defn img-data [n]
  (map mikera.image.colours/values-rgb
       (img/get-pixels
        (clisk/image [clisk/x clisk/y 0] :width 2 :height 2)))
  )

(def as-normals (comp (partial map mikera.image.colours/values-rgb) img/get-pixels))

(img/show
 (clisk/image
  (clisk/fractal
   :update (clisk/v+ clisk/x [clisk/x clisk/y])
   :bailout-result clisk/black
   :max-iterations clisk/black
   )
  :width 200 :height 200))


(clisk/show
 (clisk/viewport
  [-2 -1.5] [1 1.5]
  (clisk/fractal
   :while (clisk/v- 2 (clisk/length [clisk/x clisk/y]))
   :update (clisk/v+ clisk/c [(clisk/v- (clisk/v* clisk/x clisk/x) (clisk/v* clisk/y clisk/y)) (clisk/v* 2 clisk/x clisk/y)])
   :result (clisk/vplasma (clisk/v* 0.1 'i))
   :bailout-result clisk/black
   :max-interations 1000
   )) 512 512)

(def training-set
  [{:x [1] :y [2]}])
(layers/input 256 256 3 :id :x)

(def encoder-decoder
  (network/linear-network
    [(layers/input 256 256 3 :id :x)
     (layers/convolutional {:kernel-dim 4 :pad 2 :stride 1 :num-kernels 64})
     (layers/batch-normalization :epsilon 1e-5)

     (layers/linear->tanh 10)]))

(def trained
  (train/train-n pix2pix training-set test-set
                 :batch-size 1000
                 :network-filestem "my-fn"
                 :epoch-count 3000))
