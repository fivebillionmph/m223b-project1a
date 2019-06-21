(function() {
	var canvas = document.getElementById("app-canvas");
	canvas.setAttribute("width", "800px");
	canvas.setAttribute("height", "600px");

	var app = new Vue({
		el: "#app",
		delimiters: ["${", "}"],
		data: {
			images: [],
			selected_image: null,
			selected_image_data: null,
			brush_radius: 1,
			brush_mode: "paint",
		},
		methods: {
			refreshImages: function() {
				var that = this;
				axios.get("/api/images")
					.then(function(response) {
						that.images = response.data;
					});
			},
			sqlIntToString: function(i) {
				if(i == 1) {
					return "true";
				}
				return "false";
			},
			selectImage: function(image) {
				var that = this;

				this.selected_image = image;
				this.selected_image_data = new Image();
				this.selected_image_data.src = "/image/" + this.selected_image.name;

				if(this.selected_image.feature_mask != null) {
					this.selected_image.full_mask = decodeMask(this.selected_image.feature_mask);
				} else {
					this.selected_image.full_mask = newFullMask();
				}

				this.selected_image_data.onload = function() {
					drawCanvas(that.selected_image_data, that.selected_image.full_mask);
				};
			},
			saveMask: function(image) {
				var that = this;

				var new_feature_mask = encodeMask(image.full_mask);
				axios.post("/api/save_mask", {
					"name": image.name,
					"feature_mask": new_feature_mask,
				}).then(function() {
					image.feature_mask = new_feature_mask;
				}, function() {
					alert("couldn't update");
				});
			},
			deleteMask: function(image) {
				image.full_mask = newFullMask();
				drawCanvas(this.selected_image_data, this.selected_image.full_mask);
			},
			toggleTrainable: function(image) {
				var that = this;

				var new_trainable = image.trainable == 1 ? 0 : 1;
				axios.post("/api/set_trainable", {
					"name": image.name,
					"trainable": new_trainable,
				}).then(function() {
					image.trainable = new_trainable;
				}, function() {
					alert("couldn't update");
				});
			},
			canvasClick: function(x, y) {
				var width = this.selected_image_data.width;
				var height = this.selected_image_data.height;
				var radius = this.brush_radius;

				var left = x - radius;
				if(left < 0) left = 0;
				var right = x + radius;
				if(right > width) right = width;
				var top = y - radius;
				if(top < 0) top = 0;
				var bottom = y + radius;
				if(bottom > height) bottom = height;

				for(var i = left; i <= right; i++) {
					for(var j = top; j <= bottom; j++) {
						if(this.brush_mode == "paint") {
							this.selected_image.full_mask.points[[i,j]] = [i,j];
						} else if(this.brush_mode == "erase") {
							delete this.selected_image.full_mask.points[[i,j]];
						}
					}
				}

				drawCanvas(this.selected_image_data, this.selected_image.full_mask);
			},
			updateBrushSize: function() {
				this.brush_radius = parseInt(this.brush_radius);
			},
			clearCanvas: function() {
				clearCanvas();
			},
		},
		beforeMount: function() {
			/* initialize images */
			this.refreshImages();
		},
	});

	function encodeMask(m) {
		return JSON.stringify(m);
	}

	function decodeMask(sm) {
		return JSON.parse(sm);
	}

	function drawCanvas(image_data, mask) {
		var canvas = document.getElementById("app-canvas");
		var ctx = canvas.getContext("2d");
		var pixel = ctx.createImageData(1,1);
		pixel.data[0] = 255;
		pixel.data[1] = 0;
		pixel.data[2] = 0;
		pixel.data[3] = 255;
		ctx.drawImage(image_data, 0, 0);
		for(var point_name in mask.points) {
			var point = mask.points[point_name];
			ctx.putImageData(pixel, point[0], point[1]);
		}
	}

	(function() {
		var canvas = document.getElementById("app-canvas");
		var ctx = canvas.getContext("2d");
		var mouse_down = false;

		canvas.addEventListener("mousedown", function(e) {
			mouse_down = true;
			drawCircle(canvas, e);
		}, false);

		canvas.addEventListener("mouseup", function(e) {
			mouse_down = false;
		}, false);

		canvas.addEventListener("mousemove", function(e) {
			if(!mouse_down) {
				return;
			}
			drawCircle(canvas, e);
		}, false);

		function drawCircle(canvas, e) {
			var pos = getMousePos(canvas, e);

			app.canvasClick(pos.x, pos.y);
		}
	}());

	function getMousePos(canvas, evt) {
		/* this function taken from: https://stackoverflow.com/a/17130415/3704042 */
		var rect = canvas.getBoundingClientRect(), // abs. size of element
				scaleX = canvas.width / rect.width, // relationship bitmap vs. element for X
				scaleY = canvas.height / rect.height; // relationship bitmap vs. element for Y
	
		return {
			x: (evt.clientX - rect.left) * scaleX, // scale mouse coordinates after they have
			y: (evt.clientY - rect.top) * scaleY // been adjusted to be relative to element
		}
	}

	function newFullMask() {
		return {
			points: {},
		};
	}

}());
