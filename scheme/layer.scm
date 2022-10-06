; (define (rmelem str pos) 
;     (string-join (remove (list-ref (string-split str) (- pos 1)) (string-split str)))) 

(define (layer)
	(let* ((filelist (cadr (file-glob "*.xcf" 1))))
		(while (not (null? filelist))
            (let* (;variaveis locais
                    (filename (car filelist))
                    (only_filename (substring filename 0 (- (string-length filename) 4)))
                    (filename_mask (string-append "mask_" only_filename))                    
                    (filename_mask (string-append filename_mask ".png"))
                    (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                    (image_mask (car (gimp-file-load RUN-NONINTERACTIVE filename_mask filename_mask)))
                    ; (pos (car (gimp-image-get-item-position image_mask layer)))
                    (width (car (gimp-image-width image)))
                    (height (car (gimp-image-height image)))
                    (layer (car (gimp-image-get-layer-by-name image "Fundo")))
                    (layer_mask (car (gimp-file-load-layer RUN-NONINTERACTIVE image filename_mask)))

                    (drawable (car (gimp-image-active-drawable image)))
                )
                (gimp-item-set-name layer "exsiccata")
                (gimp-item-set-name layer_mask "mask")
                (gimp-layer-set-opacity layer_mask 60)
                (gimp-message filename)
                (gimp-image-insert-layer image layer_mask 0 -1)
                (gimp-layer-scale layer_mask width height TRUE)
                (gimp-xcf-save RUN-NONINTERACTIVE image drawable filename filename)
                (gimp-image-delete image)
                (gimp-image-delete image_mask)
            )
            (set! filelist (cdr filelist))
        )
	)
)
(script-fu-register "batch-resize"
	""
	"Do nothing"
	"Joey User"
	"Joey User"
	"August 2000"
	""
)