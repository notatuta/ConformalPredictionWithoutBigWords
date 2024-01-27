# To run,
#
# manim -pqk --disable_caching ConformalPredictionWithoutBigWords.py ConformalPredictionWithoutBigWords
#
from manim import *
import numpy as np
from scipy.stats import rankdata
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

class ConformalPredictionWithoutBigWords(MovingCameraScene):
    def construct(self):
        week = range(86)
        forecast = [32,22,33,33,36,33,55,28,49,44,66,52,55,67,52,69,59,53,68,73,73,72,82,77,80,82,87,81,87,92,87,92,83,82,86,82,85,79,77,63,58,60,64,66,57,44,47,50,45,41,27,58,42,52,49,47,33,43,52,52,45,46,44,50,65,75,62,66,59,63,67,72,88,74,81,81,84,74,86,90,80,83,83,77,83,77]
        actuals = [30,24,36,31,37,25,55,36,50,56,74,62,47,66,54,79,59,58,74,75,66,70,79,81,88,81,91,83,86,94,85,89,85,90,87,79,80,75,63,62,58,68,66,70,69,40,57,58,39,44,28,55,45,38,44,42,40,52,57,50,46,52,38,52,57,84,64,60,57,73,69,68,87,72,77,71,93,88,88,92,77,85,78,84,84,75]
        title = Text("Conformal Prediction\nwithout Big Words",font_size=48)
        self.add_sound("ConformalPredictionWithoutBigWords.mp3")
        self.play(FadeIn(title, shift=DOWN, scale=0.66))
        self.wait()
        self.play(FadeOut(title, shift=DOWN * 2, scale=1.5))
        problem_statement = Text("A seven day weather forecast says\nit's going to be 92°F.\n\nWhat is the 95% confidence interval\naround this point estimate?",font_size=36)
 #       self.add_sound("ConformalPredictionWithoutBigWords.mp3")
        self.play(FadeIn(problem_statement))
        self.wait(18)
        question_1 = Text("[87, 97]?", color=YELLOW).shift(RIGHT * 2.5).shift(UP * 0.3)
        question_2 = Text("[72, 112]?", color=YELLOW).shift(RIGHT * 2.5).shift(UP * 0.3)
        self.play(FadeIn(question_1))
        self.wait(2)
        self.play(ReplacementTransform(question_1,question_2))
        self.wait(2)
        self.play(FadeOut(question_2))
        self.wait(7)
        self.play(FadeOut(problem_statement))
        ax = Axes(
            x_range=[0, 87, 1],
            y_range=[0, 120, 10],
            x_length=9,
            y_length=6,
            x_axis_config={"include_ticks": False},
            y_axis_config={"numbers_to_include": np.arange(0, 120, 10)},
            tips=False,
        )
        self.play(Create(ax))
        labels = ax.get_axis_labels(
            x_label=Text("Week").scale(0.666666), y_label=Text("°F").scale(0.666666)
        )
        self.play(Create(labels))
        forecast_line = ax.plot_line_graph(
            x_values=range(1,87), 
            y_values=forecast, 
            add_vertex_dots=False,
            line_color=BLUE
        )
        self.play(Create(forecast_line))
        forecast_label = Text("Forecast", color=BLUE, font_size=36).next_to(ax.c2p(85, forecast[85]))
        self.play(Write(forecast_label))
        watershed_height = 115
        watershed = ax.c2p(79.5,watershed_height)
        insample_cutoff = ax.get_vertical_line(watershed)
        self.add(insample_cutoff)
        past_label = Text("Past", font_size=36).next_to(watershed, DL)
        self.play(Write(past_label))
        future_label = Text("Future", font_size=36).next_to(watershed, DR)
        self.play(Write(future_label))
        actuals_points = ax.plot_line_graph(
            x_values=range(1,80), 
            y_values=actuals[0:79], 
            add_vertex_dots=True, 
            line_color=BLACK,
            vertex_dot_style=dict(stroke_color=GREEN,  fill_color=GREEN),
            vertex_dot_radius=0.04
        )
        self.wait(2)
        self.play(Create(actuals_points["vertex_dots"]), FadeOut(labels))
        actuals_label = Text("Actuals", color=GREEN, font_size=36).next_to(ax.c2p(31, forecast[30]), UR)
        self.play(Write(actuals_label), FadeOut(past_label), FadeOut(future_label))
        self.wait(5)
        residuals_group = VGroup()
        for i in range(0,79):
            a = Dot(ax.c2p(i + 1, actuals[i]))
            b = Dot(ax.c2p(i + 1, forecast[i]))
            residuals_group.add(Line(a.get_center(), b.get_center(), color=RED))
        self.play(Create(residuals_group), FadeOut(actuals_label))
        residuals_label = Text("Differences", color=RED, font_size=36).next_to(residuals_group[45], LEFT)
        self.wait(3)
        self.play(Write(residuals_label))
        self.wait(10)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(DOWN).scale(1.15))
        residuals = np.subtract(actuals[0:79], forecast[0:79])
        downshift = 25
        self.play(*[residuals_group[i].animate.shift(
            ax.c2p(i + 1, residuals[i]) - 
            ax.c2p(i + 1, actuals[i] + downshift)
        ) for i in range(0,79)], FadeOut(residuals_label))
        new_order = (rankdata(residuals, method='ordinal') - 1).astype(int) # -1 because rank is 1-based 
        self.play(*[residuals_group[i].animate.shift(
            ax.c2p(new_order[i], 0) - 
            ax.c2p(i, 0)
        ) for i in range(0,79)])
        self.wait(5)
        residuals_lo = residuals_group[np.where(new_order==1)[0][0]]
        residuals_hi = residuals_group[np.where(new_order==77)[0][0]]
        residuals_lo_label = Text("2.5th percentile", color=WHITE, font_size=36).next_to(residuals_lo, UR)
        residuals_hi_label = Text("97.5th percentile", color=WHITE, font_size=36).next_to(residuals_hi, DL)
        residuals_lo_lead = Line(residuals_lo.get_corner(UR), residuals_lo_label.get_corner(DL), stroke_width=1)
        residuals_hi_lead = Line(residuals_hi_label.get_corner(UR), residuals_hi.get_corner(DL), stroke_width=1)
        self.play(Create(residuals_lo_lead))
        self.play(Write(residuals_lo_label))
        self.play(Write(residuals_hi_label))
        self.play(Create(residuals_hi_lead))
        lcl = sorted(residuals)[1]
        ucl = sorted(residuals)[77]
        forecast_polygon = Polygon(ax.c2p(79+1, forecast[79]+ucl, 0),
            ax.c2p(79+1, forecast[79]+lcl, 0),
            ax.c2p(80+1, forecast[80]+lcl, 0),
            ax.c2p(81+1, forecast[81]+lcl, 0),
            ax.c2p(82+1, forecast[82]+lcl, 0),
            ax.c2p(83+1, forecast[83]+lcl, 0),
            ax.c2p(84+1, forecast[84]+lcl, 0),
            ax.c2p(85+1, forecast[85]+lcl, 0),
            ax.c2p(85+1, forecast[85]+ucl, 0),
            ax.c2p(84+1, forecast[84]+ucl, 0),
            ax.c2p(83+1, forecast[83]+ucl, 0),
            ax.c2p(82+1, forecast[82]+ucl, 0),
            ax.c2p(81+1, forecast[81]+ucl, 0),
            ax.c2p(80+1, forecast[80]+ucl, 0),
            color=BLUE_D, fill_color=BLUE_D, fill_opacity=0.5
        )
        forecast_polygon.set_stroke(width=0)
        ucl_label = Text("95% UCL", color=BLUE, font_size=36).next_to(ax.c2p(85+1, forecast[85]+ucl))
        lcl_label = Text("95% LCL", color=BLUE, font_size=36).next_to(ax.c2p(85+1, forecast[85]+lcl))
        self.play(residuals_lo.animate.move_to(ax.c2p(79+1, forecast[79]+0.5*lcl)), FadeOut(residuals_lo_lead))
        self.play(residuals_hi.animate.move_to(ax.c2p(79+1, forecast[79]+0.5*ucl)), FadeOut(residuals_hi_lead))
        self.wait()
        aa = residuals_group[np.where(new_order==77)[0][0]]
        bb = residuals_group[np.where(new_order==1)[0][0]]
        self.play(Restore(self.camera.frame), 
            FadeOut(residuals_group - aa - bb),
            FadeOut(residuals_lo_label),FadeOut(residuals_hi_label)
        )
        self.play(FadeIn(forecast_polygon),
            Write(ucl_label),Write(lcl_label),
            FadeOut(residuals_lo),FadeOut(residuals_hi)
        )
        self.wait(23)
        self.play(FadeOut(forecast_label),FadeOut(forecast_line),FadeOut(forecast_polygon),
            FadeOut(lcl_label),FadeOut(ucl_label)
        )

# Part 2

        self.wait(3)
        poly_0 = lagrange(range(0,10), actuals[0:10])
        poly_1 = lagrange(range(0,10), actuals[8:18])
        poly_2 = lagrange(range(0,10), actuals[16:26])
        poly_3 = lagrange(range(0,10), actuals[24:34])
        poly_4 = lagrange(range(0,10), actuals[32:42])
        poly_5 = lagrange(range(0,10), actuals[40:50])
        poly_6 = lagrange(range(0,10), actuals[48:58])
        poly_7 = lagrange(range(0,10), actuals[56:66])
        poly_8 = lagrange(range(0,10), actuals[64:74])
        poly_9 = lagrange(range(0,10), actuals[72:82])
        def piecewise_lagrange(x):
            if x < 8+1:
                return Polynomial(poly_0.coef[::-1])(x-1)
            elif x < 16+1:
                return Polynomial(poly_1.coef[::-1])(x-1 - 8)
            elif x < 24+1:
                return Polynomial(poly_2.coef[::-1])(x-1 - 16)
            elif x < 32+1:
                return Polynomial(poly_3.coef[::-1])(x-1 - 24)
            elif x < 40+1:
                return Polynomial(poly_4.coef[::-1])(x-1 - 32)
            elif x < 48+1:
                return Polynomial(poly_5.coef[::-1])(x-1 - 40)
            elif x < 56+1:
                return Polynomial(poly_6.coef[::-1])(x-1 - 48)
            elif x < 64+1:
                return Polynomial(poly_7.coef[::-1])(x-1 - 56)
            elif x < 72+1:
                return Polynomial(poly_8.coef[::-1])(x-1 - 64)
            else:
                return Polynomial(poly_9.coef[::-1])(x-1 - 72)
        poly_graph = ax.plot(lambda x: piecewise_lagrange(x), color=BLUE)
        self.play(Create(poly_graph))
        self.wait(5)
        self.play(Uncreate(poly_graph))
        self.wait(2)
        splitter_cutoff = ax.get_vertical_line(ax.c2p(40.5,watershed_height))
        self.play(Create(splitter_cutoff))
        development_label = Text("Development", font_size=36).next_to(ax.c2p(40.5,20), LEFT)
        self.play(Write(development_label))
        calibration_label = Text("Calibration", font_size=36).next_to(ax.c2p(44.5,20), RIGHT)
        self.play(Write(calibration_label))
        self.play(FadeOut(actuals_points["vertex_dots"][40:80]))

        def func(x, amplitude, frequency, phase, dc):
            return amplitude * np.sin(frequency * x + phase) + dc
        popt, pcov = curve_fit(func, range(1,81), actuals[0:80], p0=[40,0.1,-1,60], bounds=([10,0.005,-5,0],[100,0.2,10,100]))
        def split_conformal_fun(x):
            return popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
        split_conformal_graph = ax.plot(lambda x: split_conformal_fun(x), color=BLUE)
        self.play(Create(split_conformal_graph))
        self.play(FadeIn(actuals_points["vertex_dots"][40:80]))

        split_conformal_residuals_group = VGroup()
        for i in range(40,79):
            a = Dot(ax.c2p(i+1, actuals[i]))
            b = Dot(ax.c2p(i+1, split_conformal_fun(i+1)))
            split_conformal_residuals_group.add(Line(a.get_center(), b.get_center(), color=RED))
        self.play(Create(split_conformal_residuals_group))
        self.wait()
        split_conformal_residuals = [0] * 39
        for i in range(40,79):
            split_conformal_residuals[i - 40] = actuals[i] - split_conformal_fun(i+1)

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(DOWN).scale(1.2))
        self.wait()
        downshift = 35
        self.play(*[split_conformal_residuals_group[i - 40].animate.shift(
            ax.c2p(i+1, split_conformal_residuals[i - 40]) - 
            ax.c2p(i+1, actuals[i] + downshift)
        ) for i in range(40,79)])
        split_conformal_new_order = (rankdata(split_conformal_residuals, method='ordinal') - 1).astype(int) # -1 because rank is 1-based 
        self.play(*[split_conformal_residuals_group[i - 40].animate.shift(
            ax.c2p(split_conformal_new_order[i - 40]+1, 0) - 
            ax.c2p(i+1 - 40, 0)
        ) for i in range(40,79)])
        split_conformal_residuals_lo = split_conformal_residuals_group[np.where(split_conformal_new_order==1)[0][0]]
        split_conformal_residuals_hi = split_conformal_residuals_group[np.where(split_conformal_new_order==37)[0][0]]
        split_conformal_lcl = sorted(split_conformal_residuals)[1]
        split_conformal_ucl = sorted(split_conformal_residuals)[37]
        split_conformal_forecast_polygon = Polygon(ax.c2p(79+1, split_conformal_fun(79+1)+split_conformal_ucl, 0),
            ax.c2p(79+1, split_conformal_fun(79+1)+split_conformal_lcl, 0),
            ax.c2p(80+1, split_conformal_fun(80+1)+split_conformal_lcl, 0),
            ax.c2p(81+1, split_conformal_fun(81+1)+split_conformal_lcl, 0),
            ax.c2p(82+1, split_conformal_fun(82+1)+split_conformal_lcl, 0),
            ax.c2p(83+1, split_conformal_fun(83+1)+split_conformal_lcl, 0),
            ax.c2p(84+1, split_conformal_fun(84+1)+split_conformal_lcl, 0),
            ax.c2p(85+1, split_conformal_fun(85+1)+split_conformal_lcl, 0),
            ax.c2p(85+1, split_conformal_fun(85+1)+split_conformal_ucl, 0),
            ax.c2p(84+1, split_conformal_fun(84+1)+split_conformal_ucl, 0),
            ax.c2p(83+1, split_conformal_fun(83+1)+split_conformal_ucl, 0),
            ax.c2p(82+1, split_conformal_fun(82+1)+split_conformal_ucl, 0),
            ax.c2p(81+1, split_conformal_fun(81+1)+split_conformal_ucl, 0),
            ax.c2p(80+1, split_conformal_fun(80+1)+split_conformal_ucl, 0),
            color=BLUE_D, fill_color=BLUE_D, fill_opacity=0.5
        )
        split_conformal_forecast_polygon.set_stroke(width=0)
        split_conformal_ucl_label = Text("90% UCL", color=BLUE, font_size=36).next_to(ax.c2p(86, split_conformal_fun(86)+split_conformal_ucl))
        split_conformal_lcl_label = Text("90% LCL", color=BLUE, font_size=36).next_to(ax.c2p(86, split_conformal_fun(86)+split_conformal_lcl))
        self.play(split_conformal_residuals_lo.animate.move_to(ax.c2p(80, split_conformal_fun(80)+0.5*split_conformal_lcl)))
        self.play(split_conformal_residuals_hi.animate.move_to(ax.c2p(80, split_conformal_fun(80)+0.5*split_conformal_ucl)))
        self.wait()
        self.play(Restore(self.camera.frame), 
            FadeOut(split_conformal_residuals_group - split_conformal_residuals_hi - split_conformal_residuals_lo)
        )
        self.wait()
        self.play(FadeIn(split_conformal_forecast_polygon),
            Write(split_conformal_ucl_label),Write(split_conformal_lcl_label),
            FadeOut(split_conformal_residuals_lo),FadeOut(split_conformal_residuals_hi)
        )
        self.wait(3)
        self.play(FadeOut(split_conformal_forecast_polygon), FadeOut(split_conformal_graph),
            FadeOut(split_conformal_ucl_label),FadeOut(split_conformal_lcl_label),
            FadeOut(splitter_cutoff),FadeOut(development_label),FadeOut(calibration_label)
        )

# Part 3
        vfold_cuts_group = VGroup()
        for i in range(0,4):
            vfold_cuts_group.add(ax.get_vertical_line(ax.c2p((i+1)*80/5+0.5,watershed_height)))
        self.play(Create(vfold_cuts_group))
        fold_start = [ 0,16,32,48,64]
        fold_stop  = [16,32,48,64,79]
        def func(x, amplitude, frequency, phase, dc):
            return amplitude * np.sin(frequency * x + phase) + dc
# Folds
        def cvplus_fun(x):
            return popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
        self.camera.frame.save_state()
        cvplus_residuals = [0] * 79
        cvplus_residuals_group = VGroup()
        cvplus_predictions = [0] * 79
        cvplus_predictions_group = VGroup()
        cvplus_pplusr = [0] * 79
        downshift = 120
        for i in range(0,5):
            self.play(FadeOut(actuals_points["vertex_dots"][(fold_start[i]):(fold_stop[i])]))
            if i == 0:
                x_values = range(16+1,80+1)
                y_values = actuals[16:80]
            elif i == 4:
                x_values = range(0+1,64+1)
                y_values = actuals[0:64]
            else:
                x_values = list(range(1,fold_start[i]+1))+list(range((fold_stop[i])+1,80))
                y_values = actuals[0:(fold_start[i])] + actuals[fold_stop[i]:79]
            popt, pcov = curve_fit(func, x_values, y_values, p0=[22,0.12,-2,60], bounds=([10,0.005,-5,0],[30,0.2,10,100]))
            cvplus_graph = ax.plot(lambda x: cvplus_fun(x), color=BLUE)
            for j in range(fold_start[i],fold_stop[i]):
                cvplus_predictions[j] = cvplus_fun(80)
                cvplus_predictions_group.add(Line(ax.c2p(80, cvplus_fun(80)), ax.c2p(80, 0), color=BLUE))
            self.play(Create(cvplus_graph))
            self.add(cvplus_predictions_group[fold_start[i]:fold_stop[i]])
            if i == 0:
                self.play(*[cvplus_predictions_group[j].animate.move_to(
                        ax.c2p(j+1, cvplus_predictions[j] / 2 - downshift)
                    ) for j in range(fold_start[i],fold_stop[i])],
                    self.camera.frame.animate.move_to(DOWN*2.5).scale(1.7)
                )
            else:
                self.play(*[cvplus_predictions_group[j].animate.move_to(
                    ax.c2p(j+1, cvplus_predictions[j] / 2 - downshift)
                ) for j in range(fold_start[i],fold_stop[i])])
            self.play(FadeIn(actuals_points["vertex_dots"][(fold_start[i]):(fold_stop[i])]))
            for j in range(fold_start[i],fold_stop[i]):
                a = Dot(ax.c2p(j+1, actuals[j]))
                b = Dot(ax.c2p(j+1, cvplus_fun(j+1)))
                cvplus_residuals[j] = actuals[j] - cvplus_fun(j+1)
                cvplus_pplusr[j] = cvplus_predictions[j] + cvplus_residuals[j]
                cvplus_residuals_group.add(Line(a.get_center(), b.get_center(), color=RED))
            self.play(FadeIn(cvplus_residuals_group[fold_start[i]:fold_stop[i]]))
            self.play(*[cvplus_residuals_group[j].animate.move_to(
                ax.c2p(j+1, cvplus_predictions[j] - downshift + cvplus_residuals[j] / 2)
            ) for j in range(fold_start[i],fold_stop[i])], FadeOut(cvplus_graph))
        for j in range(0,79):
            self.remove(cvplus_predictions_group[j])
            cvplus_predictions_group[j] = Line(ax.c2p(j+1, cvplus_pplusr[j] - downshift), ax.c2p(j+1, -downshift), color=BLUE)
            self.add(cvplus_predictions_group[j])
        self.play(FadeOut(cvplus_residuals_group))
        cvplus_new_order = (rankdata(cvplus_pplusr, method='ordinal') - 1).astype(int) # -1 because rank is 1-based 
        self.play( *[cvplus_predictions_group[i].animate.shift(
            ax.c2p(cvplus_new_order[i]+1, 0) - 
            ax.c2p(i+1, 0)
        ) for i in range(79)])
        self.wait(4)
        cvplus_prediction_hi = cvplus_predictions_group[np.where(cvplus_new_order==77)[0][0]]
        cvplus_prediction_lo = cvplus_predictions_group[np.where(cvplus_new_order==1)[0][0]]
        cvplus_ucl = sorted(cvplus_pplusr)[77]
        cvplus_lcl = sorted(cvplus_pplusr)[1]
        cvplus_ucl_label = Text("95% UCL", color=BLUE, font_size=36).next_to(ax.c2p(80, cvplus_ucl))
        cvplus_lcl_label = Text("95% LCL", color=BLUE, font_size=36).next_to(ax.c2p(80, cvplus_lcl))
        self.play(cvplus_prediction_hi.animate.move_to(ax.c2p(80, cvplus_ucl/2)))
        self.play(cvplus_prediction_lo.animate.move_to(ax.c2p(80, cvplus_lcl/2)))
        cvplus_predictions_group.remove(cvplus_prediction_hi)
        cvplus_predictions_group.remove(cvplus_prediction_lo)
        self.remove(cvplus_prediction_hi)
        cvplus_prediction_hi = Line(ax.c2p(80, cvplus_ucl), ax.c2p(80, cvplus_lcl), color=BLUE)
        self.add(cvplus_prediction_hi)
        self.play(Restore(self.camera.frame), FadeOut(cvplus_predictions_group), FadeOut(cvplus_prediction_lo))
        self.play(Create(cvplus_ucl_label),Create(cvplus_lcl_label))
        self.wait(3)
        paper = ImageMobject("jackknife_paper.png").scale(0.666666)
        self.play(FadeOut(cvplus_lcl_label), 
            FadeOut(cvplus_ucl_label),
            FadeOut(ax),
            FadeOut(actuals_points),
            FadeOut(insample_cutoff),
            FadeOut(cvplus_prediction_hi),
            FadeOut(vfold_cuts_group),
            FadeIn(paper)
        )
        self.wait(3)
