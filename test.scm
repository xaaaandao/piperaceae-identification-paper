(define (batch-resize pattern)
    (set! pattern (10))
    (print pattern)

    ; (let* 
    ;     ((filelist (cadr (file-glob pattern 1))))
    ;     (while (not (null? filelist))
    ;         (let* (
    ;                 (filename (car filelist))
    ;                 (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
    ;                 (drawable   (car (gimp-image-active-drawable image)))
    ;                 (cur-width  (car (gimp-image-width image)))
    ;                 (cur-height (car (gimp-image-height image)))
    ;                 (width      (* 0.25 cur-width))
    ;                 (height     (* 0.25 cur-height))
    ;             )
    ;             (gimp-message filename)
    ;             (gimp-image-scale-full image width height INTERPOLATION-CUBIC)
    ;             (let 
    ;                 ((nfilename (string-append "thumb_" filename)))
    ;                 (gimp-file-save RUN-NONINTERACTIVE image drawable nfilename nfilename)
    ;             )
    ;             (gimp-image-delete image)
    ;         )
    ;         (set! filelist (cdr filelist))
    ;     )
    ; )
)
(script-fu-register "batch-resize"
	""
	"Do nothing"
	"Joey User"
	"Joey User"
	"August 2000"
	""
	SF-STRING      "Text"          "Text Box"   ;a string variable)