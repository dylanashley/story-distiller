#!streamlit run
# -*- coding: ascii -*-

from album_extractor import LSTMAudioFeatureEncoder, Mean, OrderingLSTMEncoder
import mimetypes
import numpy as np
import os
import plotly.graph_objects as go
import streamlit as st
import tempfile
import torch
from typing import Callable, Optional
from utils import (
    build_template,
    default_template,
    fit_values,
    get_ordering_scores,
    get_value,
    scale,
)


def is_audio_file(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type and mime_type.startswith("audio/")


def plot(
    labels: list[str],
    narrative_essence: list[float],
    min_value: float,
    max_value: float,
    template: Optional[Callable[[float], float]] = None,
    basename: bool = False,
) -> go.Figure:
    """Draws a narrative essence plot for a fitting using plotly."""
    fig = go.Figure()
    x = np.linspace(0, 1, num=1000)
    if template is not None:
        fig.add_trace(
            go.Scatter(
                x=[scale(i, 0, 1, 1, len(labels)) for i in x],
                y=[scale(template(i), 0, 1, min_value, max_value) for i in x],
                mode="lines",
                name="Template",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(labels)) + 1,
            y=[
                scale(narrative_essence[song], 0, 1, min_value, max_value)
                for song in labels
            ],
            mode="markers+lines",
            name="Fitting",
        )
    )
    fig.update_layout(
        xaxis=dict(
            tickvals=np.arange(len(labels)) + 1,
            ticktext=[
                os.path.basename(label) if basename else label for label in labels
            ],
            tickangle=-90,
        ),
        yaxis=dict(title="Narrative Essence"),
    )
    return fig


def sdistill(
    files: list[str],
    encoder: torch.nn.Module,
    template: Callable = default_template,
    progress_callback: Callable = None,
    cache: dict[str, float] = dict(),
) -> tuple[list[float], go.Figure, float, dict[str, float]]:
    values = dict()
    tempo = dict()
    for i, filename in enumerate(files):
        if progress_callback is not None:
            progress_callback(i, filename)
        if os.path.basename(filename) not in cache:
            cache[os.path.basename(filename)] = get_value(filename, encoder)
        values[filename] = cache[os.path.basename(filename)]
    min_value = min(values.values())
    max_value = max(values.values())
    for k, v in values.items():
        values[k] = scale(v, min_value, max_value, 0, 1)
    playlist, fitting_loss = fit_values(values, template=template)
    return (
        [values[song] for song in playlist],
        plot(
            playlist,
            values,
            min_value,
            max_value,
            template,
            basename=True,
        ),
        fitting_loss,
        cache,
    )


def main():
    header_container = st.container()
    input_container = st.container()
    output_container = st.container()

    # initialize session state
    if "album_feature_encoder" not in st.session_state:
        st.session_state.album_feature_encoder = torch.load(
            "album_feature_encoder.pt", weights_only=False
        )
    if "album_ordering_encoder" not in st.session_state:
        st.session_state.album_ordering_encoder = torch.load(
            "album_ordering_encoder.pt", weights_only=False
        )
    if "cache" not in st.session_state:
        st.session_state.cache = dict()
    if "current_figure_idx" not in st.session_state:
        st.session_state.current_figure_idx = 0
    if "figures" not in st.session_state:
        st.session_state.figures = list()
    if "fitting_losses" not in st.session_state:
        st.session_state.fitting_losses = list()
    if "ordering_scores" not in st.session_state:
        st.session_state.ordering_scores = list()
    if "orderings" not in st.session_state:
        st.session_state.orderings = list()
    if "templates" not in st.session_state:
        templates = np.load("templates.npz")
        st.session_state.templates = [
            build_template(list(zip(templates["x"], templates["y"][i])))
            for i in [6, 7, 8, 9]
        ]

    # draw header
    header_container.markdown(
        """
        ### Automatic Album Sequencing Through Story Distillation

        This app demonstrates the ideas presented in [On the
        Distillation of Stories for Transferring Narrative Arcs in
        Collections of Independent
        Media](https://www.doi.org/10.1109/TPAMI.2024.3480702) by
        automatically ordering a playlist to fit prototypical templates
        common in music albums.

        *Note that by uploading a file on this app, you confirm that
        you have the right to execute the algorithms described in the
        above work on the file and grant permission to the authors of
        the above work to run these algorithms on the file for the
        purposes of sequencing the uploaded playlist.*

        ---
        """
    )

    # file uploader
    uploaded_files = input_container.file_uploader(
        "Upload Audio Files", accept_multiple_files=True
    )

    if uploaded_files:
        _, center, _ = input_container.columns([1, 1, 1])

        center.text("")  # hack to add a bit of spacing above the button

        # add a process button
        if center.button(
            "Process {} Files".format(len(uploaded_files)), use_container_width=True
        ):
            center.text("")  # hack to add a bit of spacing below the button
            if len(uploaded_files) < 3:
                input_container.error("Please upload at least three audio files.")
            else:
                # check all the uploaded files are audio files
                non_audio_files = [
                    file.name for file in uploaded_files if not is_audio_file(file)
                ]
                if non_audio_files:
                    input_container.error(
                        "The following files are not recognized as audio files: "
                        f"{', '.join(non_audio_files)}"
                    )
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_files = []
                        for file in uploaded_files:
                            file_path = os.path.join(temp_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            audio_files.append(file_path)

                        # clear previous results
                        st.session_state.current_figure_idx = 0
                        st.session_state.figures = list()
                        st.session_state.fitting_losses = list()
                        st.session_state.ordering_scores = list()
                        st.session_state.orderings = list()

                        # make progress bar
                        progress_bar = output_container.progress(0, text="Processing ")

                        # fit the default template and populate the cache
                        def progress_callback(i, filename):
                            progress_bar.progress(
                                0.75 * i / len(audio_files),
                                text=f"Processing {os.path.basename(filename)}",
                            )

                        (
                            playlist_ordering,
                            figure,
                            fitting_loss,
                            st.session_state.cache,
                        ) = sdistill(
                            audio_files,
                            st.session_state.album_feature_encoder,
                            progress_callback=progress_callback,
                            cache=st.session_state.cache,
                        )
                        st.session_state.orderings.append(playlist_ordering)
                        st.session_state.figures.append(figure)
                        st.session_state.fitting_losses.append(fitting_loss)

                        # fit the other templates
                        for i, template in enumerate(st.session_state.templates):
                            progress_bar.progress(
                                0.75
                                + 0.2 * (i + 1) / (len(st.session_state.templates) + 1),
                                text=f"Fitting Template {i + 2} of "
                                f"{len(st.session_state.templates) + 1}",
                            )
                            (
                                playlist_ordering,
                                figure,
                                fitting_loss,
                                st.session_state.cache,
                            ) = sdistill(
                                audio_files,
                                st.session_state.album_feature_encoder,
                                template,
                                cache=st.session_state.cache,
                            )
                            st.session_state.orderings.append(playlist_ordering)
                            st.session_state.figures.append(figure)
                            st.session_state.fitting_losses.append(fitting_loss)

                        # score the orderings
                        progress_bar.progress(0.95, text="Scoring Orderings")
                        st.session_state.ordering_scores = get_ordering_scores(
                            st.session_state.orderings,
                            st.session_state.album_ordering_encoder,
                        )
                        progress_bar.empty()

                        # get a good presentation order
                        presentation_order = np.flip(
                            np.argsort(st.session_state.ordering_scores)
                        )
                        st.session_state.figures = [
                            st.session_state.figures[i] for i in presentation_order
                        ]
                        st.session_state.fitting_losses = [
                            st.session_state.fitting_losses[i]
                            for i in presentation_order
                        ]
                        st.session_state.ordering_scores = [
                            st.session_state.ordering_scores[i]
                            for i in presentation_order
                        ]
                        st.session_state.orderings = [
                            st.session_state.orderings[i] for i in presentation_order
                        ]

                        # add the scores to the figures
                        for i in range(len(st.session_state.figures)):
                            st.session_state.figures[i].update_layout(
                                title=f"Template {presentation_order[i] + 1} <br><sup>Contrastive Score: {st.session_state.ordering_scores[i]:.4f}</sup>"
                            )

                        st.rerun()  # force a rerun to update the interface
        else:
            center.text("")  # hack to add a bit of spacing below the button

    # output container
    if len(st.session_state.figures) == len(st.session_state.templates) + 1:
        output_container.markdown("---")

        # direct sequencing tab
        col1, col2, col3 = st.columns([1, 5, 1])

        for _ in range(13):
            col1.write("")  # hack to center left button vertically
        if col1.button(
            "",
            icon=":material/arrow_back:",
            disabled=st.session_state.current_figure_idx == 0,
            key="template_back",
        ):
            st.session_state.current_figure_idx = max(
                0,
                st.session_state.current_figure_idx - 1,
            )
            st.rerun()

        col2.plotly_chart(
            st.session_state.figures[st.session_state.current_figure_idx],
            use_container_width=True,
        )

        for _ in range(13):
            col3.write("")  # hack to center right button vertically
        if col3.button(
            "",
            icon=":material/arrow_forward:",
            disabled=st.session_state.current_figure_idx
            == len(st.session_state.figures) - 1,
            key="template_forward",
        ):
            st.session_state.current_figure_idx = min(
                len(st.session_state.figures) - 1,
                st.session_state.current_figure_idx + 1,
            )
            st.rerun()


if __name__ == "__main__":
    main()
